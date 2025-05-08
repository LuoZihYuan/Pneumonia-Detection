from typing import Callable, Dict, List, Literal, Union
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import numpy as np

from transformers import AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.exceptions import NotFittedError

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


class HFPretrainedDataset(Dataset):
  def __init__(
    self,
    features: np.ndarray,
    labels: np.ndarray,
    pretrained_model_name_or_path: str,
    data_transforms: Callable = None,
  ):
    self.pretrained_model_name_or_path = pretrained_model_name_or_path
    self.features = features
    self.labels = labels
    self.data_transforms = data_transforms

    self._image_processor = AutoImageProcessor.from_pretrained(
      self.pretrained_model_name_or_path, use_fast=True
    )

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = self.features[idx]
    label = self.labels[idx]

    if self.data_transforms is not None:
      image = self.data_transforms(image)

    image = self._image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

    return image, label


class HFPretrainedClassifier(BaseEstimator, ClassifierMixin):
  def __init__(
    self,
    pretrained_model_name_or_path: str,
    # Core model configuration
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: Callable = None,
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
    self.pretrained_model_name_or_path = pretrained_model_name_or_path
    self.freeze_pretrained = freeze_pretrained
    self.freeze_except_layers = freeze_except_layers
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
    from transformers import AutoModelForImageClassification

    self._model = AutoModelForImageClassification.from_pretrained(
      self.pretrained_model_name_or_path,
      num_labels=(1 if self.n_classes_ == 2 else self.n_classes_),
      ignore_mismatched_sizes=True,
    )

    if self.freeze_pretrained:
      total_params = 0
      for param in self._model.parameters():
        total_params += param.numel()
        param.requires_grad = False

      for param in self._model.classifier.parameters():
        param.requires_grad = True

      if self.freeze_except_layers:
        for name, param in self._model.named_parameters():
          if any(layer_name in name for layer_name in self.freeze_except_layers):
            param.requires_grad = True

      if self.verbose:
        trainable_params = sum(
          param.numel() for param in self._model.parameters() if param.requires_grad
        )
        print(
          f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params / total_params:.2%})"
        )

    self._model.to(self._device)

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
      self._label_weights = self._label_weights.to(self.device)

    if self.verbose:
      print(f"Using class weights: {self._label_weights}")

  def _preapre_loss_function(self):
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

    if self.n_classes_ == 2:
      self._criterion = BCEWithLogitsLoss(pos_weight=self._label_weights)
    else:
      self._criterion = CrossEntropyLoss(weight=self._label_weights)

  def _get_dataloader(
    self,
    X: np.ndarray,
    y: np.ndarray,
    data_transforms: Callable = None,
    shuffle: bool = False,
  ):
    dataset = HFPretrainedDataset(
      torch.tensor(X),
      torch.tensor(y, dtype=torch.float32),
      self.pretrained_model_name_or_path,
      data_transforms=data_transforms,
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
    self._preapre_loss_function()

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

    if self.n_classes_ == 2 and self.threshold_calibration is not None:
      self._calibrate_thresholds(y_valid, self.predict_proba(X_valid))

    self.is_fitted_ = True
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
        logits = self._model(X_batch).logits
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
      outputs = self._model(X_batch, labels=y_batch)

      if self._label_weights is None:
        loss = outputs.loss
      else:
        logits = outputs.logits
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

        logits = self._model(X_batch).logits
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
    y_preds = []
    y_prob = self.predict_proba(X_test)
    thresholds = np.arange(0.0, 1.0, 0.01)
    for threshold in thresholds:
      y_pred = (y_prob >= threshold).astype(float)
      y_preds.append(y_pred)

    return y_preds

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

        y_out = self._model(X_batch).logits
        prob = torch.sigmoid(y_out)
        y_prob.append(prob.cpu())

    y_prob = torch.cat(y_prob).numpy()

    return y_prob


class ResNetPretrainedClassifier(HFPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "microsoft/resnet-18",
    # Core model configuration
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    data_transforms: Callable = None,
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
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
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
