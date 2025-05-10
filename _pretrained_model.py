from typing import Callable, Dict, List, Literal, Union
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import NotFittedError
from matplotlib.axes import Axes
from torchvision import transforms

try:
  # Check if we're in a Jupyter/IPython environment
  ipy_str = str(type(get_ipython()))  # type: ignore
  if "zmqshell" in ipy_str:
    # We're in a Jupyter notebook
    from tqdm.notebook import tqdm
  else:
    # We're in a regular IPython shell
    from tqdm import tqdm
except NameError:
  # We're in a standard Python interpreter
  from tqdm import tqdm


class TorchImageDataset(Dataset):
  def __init__(
    self,
    features: np.ndarray,
    labels: np.ndarray,
    data_transforms: Callable = None,
  ):
    self.features = features
    self.labels = labels
    self.data_transforms = data_transforms

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = self.data_transforms(self.features[idx])
    label = self.labels[idx]
    return image, label


class BaseTorchPretrainedImageClassifier(BaseEstimator, ClassifierMixin):
  def __init__(
    self,
    # Core model configuration
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: List[Callable] = None,
    # Training configuration
    random_state: int = None,
    shuffle: bool = True,
    batch_size: Union[str, int] = "auto",
    max_iter: int = 200,
    verbose: bool = False,
    # Optimization strategy
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential"
    ] = None,
    learning_rate: float = 1e-5,
    # Early stopping parameters
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    tol: float = 1e-4,
    n_iter_no_change: int = 10,
    # Threshold tuning parameters
    default_threshold: float = 0.5,
    threshold_calibration: Union[
      Literal["accuracy", "f1", "precision", "recall"], Callable
    ] = None,
    # Solve-specific parameters
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    # Scheduler-specific parameters
    factor: float = 0.1,  # Only used when lr_scheduler='reduce_on_plateau' or 'exponential'
    patience: int = 5,  # Only used when scheduler='reduce_on_plateau'
    min_lr: Union[
      List[float], float
    ] = 0,  # Only used when lr_scheduler='reduce_on_plateau'
    t_max: int = None,  # Only used when scheduler='cosine_annealing'
    step_size: int = 10,  # Only used when scheduler='step'
  ):
    self.class_weight = class_weight
    self.data_transforms = data_transforms
    self.random_state = random_state
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.max_iter = max_iter
    self.verbose = verbose
    self.solver = solver
    self.scheduler = scheduler
    self.learning_rate = learning_rate
    self.early_stopping = early_stopping
    self.validation_fraction = validation_fraction
    self.tol = tol
    self.n_iter_no_change = n_iter_no_change
    self.default_threshold = default_threshold
    self.threshold_calibration = threshold_calibration
    self.alpha = alpha
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.momentum = momentum
    self.nesterovs_momentum = nesterovs_momentum
    self.rho = rho
    self.max_fun = max_fun
    self.patience = patience
    self.factor = factor
    self.min_lr = min_lr
    self.t_max = t_max
    self.step_size = step_size

    self.is_fitted_ = False

    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._threshold = self.default_threshold

    if self.random_state is not None:
      self._rng = np.random.RandomState(self.random_state)
      torch.manual_seed(self.random_state)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
      self._rng = np.random.RandomState()

  def _prepare_labels(self, y: np.ndarray):
    unique_labels = np.sort(np.unique(y))
    if not hasattr(self, "classes_"):
      if len(unique_labels) < 2:
        raise ValueError(f"The data contains only one class {unique_labels}")
      self.classes_ = unique_labels
      self.n_classes_ = len(self.classes_)
    diff = np.setdiff1d(unique_labels, self.classes_, assume_unique=True)
    if len(diff) > 0:
      raise ValueError(f"y contains new labels {diff}")
    if np.array_equal(unique_labels, np.arange(len(self.classes_))):
      return y
    y_map = {label: i for i, label in enumerate(self.classes_)}
    y = np.array([y_map[label] for label in y])
    return y

  def _prepare_model(self):
    raise NotImplementedError(
      "Create a subclass that inherits this one and setup `self._model` in your overriden `_prepare_model` method."
    )

  def _prepare_optimizer(self):
    from torch.optim import AdamW, Adam, SGD, RMSprop, LBFGS

    parameters = filter(lambda p: p.requires_grad, self._model.parameters())

    match self.solver:
      case "adamw":
        self._optimizer = AdamW(
          params=parameters,
          lr=self.learning_rate,
          betas=(self.beta_1, self.beta_2),
          eps=self.epsilon,
          weight_decay=self.alpha,
        )
      case "adam":
        self._optimizer = Adam(
          params=parameters,
          lr=self.learning_rate,
          betas=(self.beta_1, self.beta_2),
          eps=self.epsilon,
          weight_decay=self.alpha,
        )
      case "sgd":
        self._optimizer = SGD(
          params=parameters,
          lr=self.learning_rate,
          momentum=self.momentum,
          weight_decay=self.alpha,
          nesterov=self.nesterovs_momentum,
        )
      case "rmsprop":
        self._optimizer = RMSprop(
          params=parameters,
          lr=self.learning_rate,
          alpha=self.rho,
          eps=self.epsilon,
          weight_decay=self.alpha,
          momentum=self.momentum,
        )
      case "lbfgs":
        self._optimizer = LBFGS(
          params=parameters,
          lr=self.learning_rate,
          max_eval=self.max_fun,
        )

  def _prepare_scheduler(self):
    from torch.optim.lr_scheduler import (
      ReduceLROnPlateau,
      CosineAnnealingLR,
      StepLR,
      ExponentialLR,
    )

    match self.scheduler:
      case "reduce_on_plateau":
        self._scheduler = ReduceLROnPlateau(
          self._optimizer,
          "min",
          factor=self.factor,
          patience=self.patience,
          min_lr=self.min_lr,
        )
      case "cosine_annealing":
        self._scheduler = CosineAnnealingLR(
          self._optimizer, T_max=self.t_max, eta_min=self.min_lr
        )
      case "step":
        self._scheduler = StepLR(
          self._optimizer, step_size=self.step_size, gamma=self.factor
        )
      case "exponential":
        self._scheduler = ExponentialLR(self._optimizer, gamma=self.factor)
      case None:
        self._scheduler = None

  def _prepare_weights(self, y_train: np.ndarray):
    from sklearn.utils.class_weight import compute_class_weight

    self._label_weights = None

    if self.class_weight is None:
      return
    elif self.class_weight == "balanced":
      weights = compute_class_weight(
        "balanced", classes=self.classes_, y=y_train.flatten()
      )
      class_weights_dict = {
        label: weight for label, weight in zip(self.classes_, weights)
      }
    elif isinstance(self.class_weight, dict):
      class_weights_dict = self.class_weight

    if self.n_classes_ == 2:
      pos_weight = class_weights_dict.get(1, 1.0) / class_weights_dict.get(0, 1.0)
      self._label_weights = torch.tensor([pos_weight], device=self._device)
    else:
      self._label_weights = torch.zeros(self.n_classes_)
      for label, weight in class_weights_dict.items():
        self._label_weights[label] = weight
      self._label_weights = self._label_weights.to(self._device)

    if self.verbose:
      print(f"Using class weights: {self._label_weights}")

  def _prepare_loss_function(self):
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

    if self.n_classes_ == 2:
      self._criterion = BCEWithLogitsLoss(pos_weight=self._label_weights)
    else:
      self._criterion = CrossEntropyLoss(weight=self._label_weights)

  def _get_dataloader(
    self,
    X: np.ndarray,
    y: np.ndarray,
    data_transforms: List[Callable] = None,
    shuffle: bool = False,
  ):
    default_transforms = [
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225],  # ImageNet std
      ),
    ]

    if data_transforms is not None:
      default_transforms[0:0] = data_transforms

    dataset = TorchImageDataset(
      torch.tensor(X, dtype=torch.float32),
      torch.tensor(y, dtype=torch.float32),
      data_transforms=transforms.Compose(default_transforms),
    )

    if self.batch_size == "auto":
      batch_size = min(32, len(X))
    else:
      batch_size = self.batch_size

    generator = None
    if self.random_state is not None:
      generator = torch.Generator()
      generator.manual_seed(self.random_state)

    return DataLoader(
      dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )

  def fit(self, X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.model_selection import train_test_split

    self.is_fitted_ = False

    self._prepare_labels(y_train)
    self._prepare_model()
    self._prepare_optimizer()
    self._prepare_scheduler()
    self._prepare_weights(y_train)
    self._prepare_loss_function()

    validloader = None
    if self.early_stopping or self.scheduler == "reduce_on_plateau":
      X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=self.validation_fraction,
        random_state=self.random_state,
      )
      validloader = self._get_dataloader(X_valid, y_valid)
    trainloader = self._get_dataloader(
      X_train, y_train, data_transforms=self.data_transforms, shuffle=self.shuffle
    )

    self._train_model(trainloader, validloader)
    self.is_fitted_ = True

    if self.n_classes_ == 2 and self.threshold_calibration is not None:
      self._calibrate_thresholds(y_valid, self.predict_proba(X_valid))

    return self

  def _train_model(self, trainloader: DataLoader, validloader: DataLoader = None):
    early_stopping_counter = 0
    best_model_weights = None

    self.train_loss_: float = 0
    self.best_train_loss_: float = float("inf")
    self.train_loss_curve_: List[float] = []
    self.train_history_: List = []

    if validloader is not None:
      self.validation_loss_: float = 0
      self.best_validation_loss_: float = float("inf")
      self.validation_loss_curve_: List[float] = []
      self.validation_history_: List = []

    for epoch in range(self.max_iter):
      self.n_iter_ = epoch + 1

      self._train_epoch(trainloader)
      self.train_loss_ = self.train_loss_curve_[-1]
      self.best_train_loss_ = min(self.best_train_loss_, self.train_loss_)

      y_true, y_pred = self.train_history_[-1]
      if self.verbose:
        print(
          f"(Training) Loss: {self.train_loss_:.4f}, "
          f"Accuracy: {accuracy_score(y_true, y_pred):.4f}, "
          f"F1: {f1_score(y_true, y_pred):.4f}, "
          f"Precision: {precision_score(y_true, y_pred):.4f}, "
          f"Recall: {recall_score(y_true, y_pred):.4f}"
        )

      if validloader is not None:
        self._evaluate_epoch(validloader)
        self.validation_loss_ = self.validation_loss_curve_[-1]

        y_true, y_pred = self.validation_history_[-1]
        if self.verbose:
          print(
            f"(Validation) Loss: {self.validation_loss_:.4f}, "
            f"Accuracy: {accuracy_score(y_true, y_pred):.4f}, "
            f"F1: {f1_score(y_true, y_pred):.4f}, "
            f"Precision: {precision_score(y_true, y_pred):.4f}, "
            f"Recall: {recall_score(y_true, y_pred):.4f}"
          )

        if self.scheduler == "reduce_on_plateau":
          self._scheduler.step(self.validation_loss_)

        if self.early_stopping:
          if self.validation_loss_ < self.best_validation_loss_ - self.tol:
            self.best_validation_loss_ = self.validation_loss_
            best_model_weights = self._model.state_dict().copy()
            early_stopping_counter = 0
            if self.verbose:
              print(f"-- New best validation loss: {self.best_validation_loss_:.4f} --")
          else:
            early_stopping_counter += 1
            if self.verbose:
              print(f"No improvement for {early_stopping_counter} epochs")

          if early_stopping_counter >= self.n_iter_no_change:
            if self.verbose:
              print("Early stopping triggered")
            break
        elif self.validation_loss_ < self.best_validation_loss_:
          self.best_validation_loss_ = self.validation_loss_
          best_model_weights = self._model.state_dict().copy()

      if self.scheduler in ["cosine_annealing", "step", "exponential"]:
        self._scheduler.step()
        if self.verbose:
          curr_lr = self._optimizer.param_groups[0]["lr"]
          print(f"Current learning rate: {curr_lr:.2e}")
    if best_model_weights is not None:
      self._model.load_state_dict(best_model_weights)
      if self.verbose:
        print(
          f"-- Loaded best model with validation loss: {self.best_validation_loss_:.4f} --"
        )

  def _train_epoch(self, trainloader: DataLoader):
    self._model.train()

    epoch_train_loss = 0
    y_logit = []
    y_true = []

    iterator = (
      tqdm(trainloader, desc=f"Epoch {self.n_iter_}/{self.max_iter}")
      if self.verbose
      else trainloader
    )
    for batch in iterator:
      X_batch, y_batch = batch
      X_batch = X_batch.to(self._device)
      y_batch = y_batch.unsqueeze(1).float().to(self._device)

      batch_loss = self._train_batch(X_batch, y_batch)

      epoch_train_loss += batch_loss / len(trainloader)

      with torch.no_grad():
        logits = self._model(X_batch)
        y_logit.append(logits.cpu())
        y_true.append(y_batch.cpu())

    with torch.no_grad():
      self._model.eval()
      y_prob = torch.cat(y_logit, dim=0).sigmoid().numpy()
      y_true = torch.cat(y_true, dim=0).numpy()
      y_pred = (y_prob >= self._threshold).astype(int)

    self.train_loss_curve_.append(epoch_train_loss)
    self.train_history_.append((y_true, y_pred))

  def _train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray):
    def closure():
      self._optimizer.zero_grad()
      logits = self._model(X_batch)
      loss = self._criterion(logits, y_batch)

      loss.backward()
      return loss

    if self.solver == "lbfgs":
      loss = self._optimizer.step(closure)
    else:
      loss = closure()
      self._optimizer.step()

    return loss.item()

  def _evaluate_epoch(self, validloader):
    self._model.eval()

    y_true = []
    y_prob = []
    epoch_validation_loss = 0
    with torch.no_grad():
      for X_batch, y_batch in validloader:
        X_batch = X_batch.to(self._device)
        y_batch = y_batch.unsqueeze(1).float().to(self._device)

        logits = self._model(X_batch)
        loss = self._criterion(logits, y_batch)
        epoch_validation_loss += loss.item() / len(validloader)

        probs = torch.sigmoid(logits)
        y_prob.append(probs.cpu())
        y_true.append(y_batch.cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    y_pred = (y_prob >= self._threshold).astype(int)

    self.validation_loss_curve_.append(epoch_validation_loss)
    self.validation_history_.append((y_true, y_pred))

  def _calibrate_thresholds(self, y_valid: np.ndarray, y_prob: np.ndarray):
    if callable(self.threshold_calibration):
      scorer = self.threshold_calibration
    elif isinstance(self.threshold_calibration, str):
      match self.threshold_calibration:
        case "f1":
          scorer = f1_score
        case "precision":
          scorer = precision_score
        case "recall":
          scorer = recall_score
        case "accuracy":
          scorer = accuracy_score

    best_score = 0
    best_threshold = self.default_threshold
    thresholds = np.arange(0.0, 1.0, 0.01)
    for threshold in thresholds:
      y_pred = (y_prob >= threshold).astype(float)
      score = scorer(y_valid, y_pred)

      if score > best_score:
        best_score = score
        best_threshold = threshold
    if self.verbose:
      print(f"Best threshold: {best_threshold:.2f} with best score: {best_score:.4f}")
    self._threshold = best_threshold

  def predict(self, X_test: np.ndarray):
    if not self.is_fitted_:
      msg = (
        f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
      )
      raise NotFittedError(msg)

    y_prob = self.predict_proba(X_test)
    y_pred = (y_prob >= self._threshold).astype(float)

    return y_pred

  def predict_proba(self, X_test: np.ndarray):
    if not self.is_fitted_:
      msg = (
        f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
      )
      raise NotFittedError(msg)

    testloader = self._get_dataloader(X_test, np.zeros(len(X_test)))
    self._model.eval()

    y_prob = []
    with torch.no_grad():
      for X_batch, _ in testloader:
        X_batch = X_batch.to(self._device)
        logits = self._model(X_batch)

        if self.n_classes_ == 2:
          prob = torch.sigmoid(logits)
        else:
          prob = torch.softmax(logits, dim=1)

        y_prob.append(prob.cpu())

    y_prob = torch.cat(y_prob).numpy()

    return y_prob

  def plot(
    self,
    X_test: np.ndarray,
    target_classes: List[int] = None,
    method_name: Literal[
      "gradcam",
      "finercam",
      "shapleycam",
      "fem",
      "hirescam",
      "gradcamelementwise",
      "ablationcam",
      "xgradcam",
      "gradcamplusplus",
      "scorecam",
      "eigencam",
      "eigengradcam",
      "kpca_cam",
      "randomcam",
      "fullgrad",
    ] = "gradcam",
    ax: Union[np.ndarray, Axes] = None,
    **kwargs,
  ):
    from matplotlib import pyplot as plt
    from pytorch_grad_cam import (
      GradCAM,
      FinerCAM,
      ShapleyCAM,
      FEM,
      HiResCAM,
      GradCAMElementWise,
      AblationCAM,
      XGradCAM,
      GradCAMPlusPlus,
      ScoreCAM,
      EigenCAM,
      EigenGradCAM,
      KPCA_CAM,
      RandomCAM,
      FullGrad,
    )
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import (
      BinaryClassifierOutputTarget,
      ClassifierOutputTarget,
    )

    method_map = {
      "gradcam": GradCAM,
      "finercam": FinerCAM,
      "shapleycam": ShapleyCAM,
      "fem": FEM,
      "hirescam": HiResCAM,
      "gradcamelementwise": GradCAMElementWise,
      "ablationcam": AblationCAM,
      "xgradcam": XGradCAM,
      "gradcamplusplus": GradCAMPlusPlus,
      "scorecam": ScoreCAM,
      "eigencam": EigenCAM,
      "eigengradcam": EigenGradCAM,
      "kpca_cam": KPCA_CAM,
      "randomcam": RandomCAM,
      "fullgrad": FullGrad,
    }
    method = method_map.get(method_name)

    if target_classes is None:
      plot_classes = self.predict(X_test).reshape((-1, 1))
    else:
      plot_classes = np.stack((target_classes,) * len(X_test))

    if ax is None:
      _, ax = plt.subplots(
        plot_classes.shape[1],
        plot_classes.shape[0],
        figsize=np.array(plot_classes.shape) * 6.5,
      )
    if type(ax).__name__ == "ndarray" and ax.ndim == 1:
      ax = ax.reshape(np.flip(plot_classes.shape))
    elif type(ax).__name__ == "Axes":
      ax = np.array([[ax]])

    if not np.array_equal(ax.shape, np.flip(plot_classes.shape)):
      raise ValueError(f"shape of ax should be {plot_classes.shape}")

    data_transforms = transforms.Compose(
      [
        transforms.Normalize(
          mean=[0.485, 0.456, 0.406],  # ImageNet mean
          std=[0.229, 0.224, 0.225],  # ImageNet std
        ),
      ]
    )
    dataset = TorchImageDataset(
      torch.tensor(X_test, dtype=torch.float32), np.zeros(len(X_test)), data_transforms
    )
    self._model.eval()

    with method(model=self._model, target_layers=self._gradcam_layers) as cam:
      for i, row in enumerate(plot_classes):
        background = np.transpose(X_test[i], (1, 2, 0)).astype(np.float32) / 255.0
        for j, plot_class in enumerate(row):
          image = dataset[i][0].unsqueeze(0).to(self._device)
          targets = [
            BinaryClassifierOutputTarget(plot_class)
            if self.n_classes_ == 2
            else ClassifierOutputTarget(plot_class)
          ]
          mask = cam(image, targets)[0, :]
          visualization = show_cam_on_image(
            background, mask, use_rgb=True, image_weight=0.75
          )
          ax[j, i].imshow(visualization)
          ax[j, i].set_title(f"{type(self._model).__name__} (Class: {plot_class})")
          ax[j, i].axis("off")

    return ax


class ConvNeXtPretrainedClassifier(BaseTorchPretrainedImageClassifier):
  def __init__(
    self,
    architecture: Literal[
      "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
    ] = "convnext_tiny",
    # Core model configuration
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: List[Callable] = None,
    # Training configuration
    random_state: int = None,
    shuffle: bool = True,
    batch_size: Union[str, int] = "auto",
    max_iter: int = 200,
    verbose: bool = False,
    # Optimization strategy
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential"
    ] = "reduce_on_plateau",
    learning_rate: float = 0.001,
    # Early stopping parameters
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    tol: float = 1e-4,
    n_iter_no_change: int = 10,
    # Threshold tuning parameters
    default_threshold: float = 0.5,
    threshold_calibration: Union[
      Literal["accuracy", "f1", "precision", "recall"], Callable
    ] = None,
    # Solve-specific parameters
    alpha: float = 0,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    # Scheduler-specific parameters
    factor: float = 0.5,  # Only used when lr_scheduler='reduce_on_plateau' or 'exponential'
    patience: int = 2,  # Only used when scheduler='reduce_on_plateau'
    min_lr: Union[
      List[float], float
    ] = 0,  # Only used when lr_scheduler='reduce_on_plateau'
    t_max: int = None,  # Only used when scheduler='cosine_annealing'
    step_size: int = 10,  # Only used when scheduler='step'
  ):
    super().__init__(
      class_weight=class_weight,
      data_transforms=data_transforms,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      verbose=verbose,
      solver=solver,
      scheduler=scheduler,
      learning_rate=learning_rate,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      default_threshold=default_threshold,
      threshold_calibration=threshold_calibration,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      factor=factor,
      patience=patience,
      min_lr=min_lr,
      t_max=t_max,
      step_size=step_size,
    )
    self.architecture = architecture
    self.freeze_pretrained = freeze_pretrained
    self.freeze_except_layers = freeze_except_layers

  def _prepare_model(self):
    from torchvision.models import (
      convnext_tiny,
      convnext_small,
      convnext_base,
      convnext_large,
    )

    architecture_map = {
      "convnext_tiny": convnext_tiny,
      "convnext_small": convnext_small,
      "convnext_base": convnext_base,
      "convnext_large": convnext_large,
    }

    model_fn = architecture_map.get(self.architecture)
    self._model = model_fn(weights="IMAGENET1K_V1")
    num_features = self._model.classifier[-1].in_features
    if self.n_classes_ == 2:
      self._model.classifier[-1] = torch.nn.Linear(num_features, 1)
    else:
      self._model.classifier[-1] = torch.nn.Linear(num_features, self.n_classes_)

    if self.freeze_pretrained:
      total_params = 0
      trainable_params = 0

      for name, param in self._model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False

        if "classifier.2" in name:
          param.requires_grad = True
          trainable_params += param.numel()

        if self.freeze_except_layers is not None:
          for layer_name in self.freeze_except_layers:
            if layer_name in name:
              param.requires_grad = True
              trainable_params += param.numel()
              # Avoid double counting
              if "classifier.2" in name:
                trainable_params -= param.numel()

      if self.verbose:
        print(
          f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.2%})"
        )

    self._gradcam_layers = [self._model.features[7][-1].block]
    self._model.to(self._device)


class DenseNetPretrainedClassifier(BaseTorchPretrainedImageClassifier):
  def __init__(
    self,
    architecture: Literal[
      "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
    ] = "convnext_tiny",
    # Core model configuration
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: List[Callable] = None,
    # Training configuration
    random_state: int = None,
    shuffle: bool = True,
    batch_size: Union[str, int] = "auto",
    max_iter: int = 200,
    verbose: bool = False,
    # Optimization strategy
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential"
    ] = "reduce_on_plateau",
    learning_rate: float = 0.001,
    # Early stopping parameters
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    tol: float = 1e-4,
    n_iter_no_change: int = 10,
    # Threshold tuning parameters
    default_threshold: float = 0.5,
    threshold_calibration: Union[
      Literal["accuracy", "f1", "precision", "recall"], Callable
    ] = None,
    # Solve-specific parameters
    alpha: float = 0,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    # Scheduler-specific parameters
    factor: float = 0.5,  # Only used when lr_scheduler='reduce_on_plateau' or 'exponential'
    patience: int = 2,  # Only used when scheduler='reduce_on_plateau'
    min_lr: Union[
      List[float], float
    ] = 0,  # Only used when lr_scheduler='reduce_on_plateau'
    t_max: int = None,  # Only used when scheduler='cosine_annealing'
    step_size: int = 10,  # Only used when scheduler='step'
  ):
    super().__init__(
      class_weight=class_weight,
      data_transforms=data_transforms,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      verbose=verbose,
      solver=solver,
      scheduler=scheduler,
      learning_rate=learning_rate,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      default_threshold=default_threshold,
      threshold_calibration=threshold_calibration,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      factor=factor,
      patience=patience,
      min_lr=min_lr,
      t_max=t_max,
      step_size=step_size,
    )
    self.architecture = architecture
    self.freeze_pretrained = freeze_pretrained
    self.freeze_except_layers = freeze_except_layers

  def _prepare_model(self):
    from torchvision.models import densenet121, densenet161, densenet169, densenet201

    architecture_map = {
      "densenet121": densenet121,
      "densenet161": densenet161,
      "densenet169": densenet169,
      "densenet201": densenet201,
    }

    model_fn = architecture_map.get(self.architecture)
    self._model = model_fn(weights="IMAGENET1K_V1")
    num_features = self._model.classifier.in_features
    if self.n_classes_ == 2:
      self._model.classifier = torch.nn.Linear(num_features, 1)
    else:
      self._model.classifier = torch.nn.Linear(num_features, self.n_classes_)

    if self.freeze_pretrained:
      total_params = 0
      trainable_params = 0

      for name, param in self._model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False

        if "classifier" in name:
          param.requires_grad = True
          trainable_params += param.numel()

        if self.freeze_except_layers is not None:
          for layer_name in self.freeze_except_layers:
            if layer_name in name:
              param.requires_grad = True
              trainable_params += param.numel()
              # Avoid double counting
              if "classifier" in name:
                trainable_params -= param.numel()

      if self.verbose:
        print(
          f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.2%})"
        )

    self._gradcam_layers = [self._model.features[-1]]
    self._model.to(self._device)


class ResNetPretrainedClassifier(BaseTorchPretrainedImageClassifier):
  def __init__(
    self,
    architecture: Literal[
      "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ] = "resnet18",
    # Core model configuration
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: List[Callable] = None,
    # Training configuration
    random_state: int = None,
    shuffle: bool = True,
    batch_size: Union[str, int] = "auto",
    max_iter: int = 200,
    verbose: bool = False,
    # Optimization strategy
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential"
    ] = "reduce_on_plateau",
    learning_rate: float = 0.001,
    # Early stopping parameters
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    tol: float = 1e-4,
    n_iter_no_change: int = 10,
    # Threshold tuning parameters
    default_threshold: float = 0.5,
    threshold_calibration: Union[
      Literal["accuracy", "f1", "precision", "recall"], Callable
    ] = None,
    # Solve-specific parameters
    alpha: float = 0,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    # Scheduler-specific parameters
    factor: float = 0.5,  # Only used when lr_scheduler='reduce_on_plateau' or 'exponential'
    patience: int = 2,  # Only used when scheduler='reduce_on_plateau'
    min_lr: Union[
      List[float], float
    ] = 0,  # Only used when lr_scheduler='reduce_on_plateau'
    t_max: int = None,  # Only used when scheduler='cosine_annealing'
    step_size: int = 10,  # Only used when scheduler='step'
  ):
    super().__init__(
      class_weight=class_weight,
      data_transforms=data_transforms,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      verbose=verbose,
      solver=solver,
      scheduler=scheduler,
      learning_rate=learning_rate,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      default_threshold=default_threshold,
      threshold_calibration=threshold_calibration,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      factor=factor,
      patience=patience,
      min_lr=min_lr,
      t_max=t_max,
      step_size=step_size,
    )
    self.architecture = architecture
    self.freeze_pretrained = freeze_pretrained
    self.freeze_except_layers = freeze_except_layers

  def _prepare_model(self):
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

    architecture_map = {
      "resnet18": resnet18,
      "resnet34": resnet34,
      "resnet50": resnet50,
      "resnet101": resnet101,
      "resnet152": resnet152,
    }

    model_fn = architecture_map.get(self.architecture)
    self._model = model_fn(weights="IMAGENET1K_V1")
    num_features = self._model.fc.in_features
    if self.n_classes_ == 2:
      self._model.fc = torch.nn.Linear(num_features, 1)
    else:
      self._model.fc = torch.nn.Linear(num_features, self.n_classes_)

    if self.freeze_pretrained:
      total_params = 0
      trainable_params = 0

      for name, param in self._model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False

        if "fc" in name:
          param.requires_grad = True
          trainable_params += param.numel()

        if self.freeze_except_layers is not None:
          for layer_name in self.freeze_except_layers:
            if layer_name in name:
              param.requires_grad = True
              trainable_params += param.numel()
              # Avoid double counting
              if "fc" in name:
                trainable_params -= param.numel()

      if self.verbose:
        print(
          f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.2%})"
        )

    self._gradcam_layers = [self._model.layer4[-1]]
    self._model.to(self._device)
