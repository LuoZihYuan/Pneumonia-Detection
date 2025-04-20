from typing import Literal, Dict, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD, RMSprop, LBFGS
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
  ReduceLROnPlateau,
  CosineAnnealingLR,
  StepLR,
  ExponentialLR,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoImageProcessor, AutoModelForImageClassification


class HGPretrainedClassifier(BaseEstimator, ClassifierMixin):
  """
  A scikit-learn compatible classifier that uses pre-trained HuggingFace vision models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str,
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True,
    rho: float = 0.99,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    """
    Initialize the classifier with the given parameters.

    Args:
        pretrained_model_name_or_path: Name or path of the pre-trained model
        binary_classification: If True, perform binary classification
        freeze_pretrained: If True, freeze all layers except the classifier
        freeze_except_layers: List of layer names to unfreeze when freeze_pretrained is True
        class_weight: Class weights for imbalanced datasets
        solver: Optimizer to use for training
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle the training data
        batch_size: Batch size for training, or "auto" to determine automatically
        max_iter: Maximum number of training epochs
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of training data to use for validation
        tol: Tolerance for early stopping
        n_iter_no_change: Number of epochs with no improvement for early stopping
        learning_rate: Learning rate for the optimizer
        alpha: L2 penalty (regularization term) parameter
        beta_1: Exponential decay rate for first moment estimates (Adam/AdamW)
        beta_2: Exponential decay rate for second moment estimates (Adam/AdamW)
        epsilon: Value for numerical stability (Adam/AdamW/RMSprop)
        momentum: Momentum factor (SGD)
        nesterovs_momentum: Whether to use Nesterov momentum (SGD)
        rho: Squared gradient rolling average factor (RMSprop)
        max_fun: Maximum number of function evaluations (LBFGS)
        lr_scheduler: Learning rate scheduler to use
        lr_scheduler_patience: Patience for ReduceLROnPlateau
        lr_scheduler_factor: Factor for reducing learning rate
        lr_scheduler_min_lr: Minimum learning rate
        lr_scheduler_t_max: Maximum number of iterations for CosineAnnealingLR
        lr_scheduler_step_size: Period of learning rate decay for StepLR
        verbose: Whether to print training progress
    """
    # Store all parameters
    self.pretrained_model_name_or_path = pretrained_model_name_or_path
    self.binary_classification = binary_classification
    self.freeze_pretrained = freeze_pretrained
    self.freeze_except_layers = freeze_except_layers or []
    self.class_weight = class_weight
    self.solver = solver
    self.random_state = random_state
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.max_iter = max_iter
    self.early_stopping = early_stopping
    self.validation_fraction = validation_fraction
    self.tol = tol
    self.n_iter_no_change = n_iter_no_change
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.momentum = momentum
    self.nesterovs_momentum = nesterovs_momentum
    self.rho = rho
    self.max_fun = max_fun
    self.lr_scheduler = lr_scheduler
    self.lr_scheduler_patience = lr_scheduler_patience
    self.lr_scheduler_factor = lr_scheduler_factor
    self.lr_scheduler_min_lr = lr_scheduler_min_lr
    self.lr_scheduler_t_max = lr_scheduler_t_max or max_iter
    self.lr_scheduler_step_size = lr_scheduler_step_size
    self.verbose = verbose

    # Initialize internal attributes
    self._model = None
    self._image_processor = None
    self._optimizer = None
    self._scheduler = None
    self._criterion = None
    self._label_weights = None
    self._best_model_state = None
    self.unique_labels = None

    # Set up random state
    if self.random_state is not None:
      self.rng = np.random.RandomState(self.random_state)
      torch.manual_seed(self.random_state)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
      self.rng = np.random.RandomState()

    # Set device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def _prepare_model(self):
    """
    Load and prepare the pre-trained model.
    """
    # Load image processor
    self._image_processor = AutoImageProcessor.from_pretrained(
      self.pretrained_model_name_or_path,
    )

    # Determine number of labels for the model
    if self.binary_classification:
      num_labels = 1
    else:
      num_labels = len(self.unique_labels)

    # Load model
    self._model = AutoModelForImageClassification.from_pretrained(
      self.pretrained_model_name_or_path,
      num_labels=num_labels,
      ignore_mismatched_sizes=True,
    )

    # Freeze layers if specified
    if self.freeze_pretrained:
      # First freeze all parameters
      for param in self._model.parameters():
        param.requires_grad = False

      # Then unfreeze the classifier layer
      for param in self._model.classifier.parameters():
        param.requires_grad = True

      # Unfreeze specific layers if requested
      if self.freeze_except_layers:
        for name, param in self._model.named_parameters():
          if any(layer_name in name for layer_name in self.freeze_except_layers):
            param.requires_grad = True

      # Print trainable parameters info
      if self.verbose:
        trainable_params = sum(
          param.numel() for param in self._model.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self._model.parameters())
        print(
          f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})"
        )
        print(f"Total parameters: {total_params:,}")

    # Move model to device
    self._model.to(self.device)

  def _prepare_optimizer(self):
    """
    Set up the optimizer based on the selected solver.
    """
    parameters = [param for param in self._model.parameters() if param.requires_grad]

    if self.solver == "adamw":
      self._optimizer = AdamW(
        parameters,
        lr=self.learning_rate,
        betas=(self.beta_1, self.beta_2),
        eps=self.epsilon,
        weight_decay=self.alpha,
      )
    elif self.solver == "adam":
      self._optimizer = Adam(
        parameters,
        lr=self.learning_rate,
        betas=(self.beta_1, self.beta_2),
        eps=self.epsilon,
        weight_decay=self.alpha,
      )
    elif self.solver == "sgd":
      self._optimizer = SGD(
        parameters,
        lr=self.learning_rate,
        momentum=self.momentum,
        weight_decay=self.alpha,
        nesterov=self.nesterovs_momentum,
      )
    elif self.solver == "rmsprop":
      self._optimizer = RMSprop(
        parameters,
        lr=self.learning_rate,
        eps=self.epsilon,
        weight_decay=self.alpha,
        momentum=self.momentum,
        alpha=self.rho,
      )
    elif self.solver == "lbfgs":
      self._optimizer = LBFGS(
        parameters,
        lr=self.learning_rate,
        max_eval=self.max_fun,
      )

  def _prepare_scheduler(self):
    """
    Set up the learning rate scheduler.
    """
    if self.lr_scheduler == "reduce_on_plateau":
      self._scheduler = ReduceLROnPlateau(
        self._optimizer,
        mode="min",
        factor=self.lr_scheduler_factor,
        patience=self.lr_scheduler_patience,
        min_lr=self.lr_scheduler_min_lr,
      )
    elif self.lr_scheduler == "cosine_annealing":
      self._scheduler = CosineAnnealingLR(
        self._optimizer, T_max=self.lr_scheduler_t_max, eta_min=self.lr_scheduler_min_lr
      )
    elif self.lr_scheduler == "step":
      self._scheduler = StepLR(
        self._optimizer,
        step_size=self.lr_scheduler_step_size,
        gamma=self.lr_scheduler_factor,
      )
    elif self.lr_scheduler == "exponential":
      self._scheduler = ExponentialLR(self._optimizer, gamma=self.lr_scheduler_factor)
    else:
      self._scheduler = None

  def _prepare_weights(self, y_train: np.ndarray):
    """
    Prepare class weights for imbalanced datasets.

    Args:
        y_train: Training labels
    """
    self._label_weights = None

    if self.class_weight is None:
      return
    elif self.class_weight == "balanced":
      weights = compute_class_weight(
        class_weight="balanced", classes=self.unique_labels, y=y_train
      )
      class_weights_dict = {
        label: weight for label, weight in zip(self.unique_labels, weights)
      }
    elif isinstance(self.class_weight, dict):
      class_weights_dict = self.class_weight
    else:
      raise ValueError(f"Unsupported class_weight parameter: {self.class_weight}")

    if self.binary_classification:
      pos_weight = class_weights_dict.get(1, 1.0) / class_weights_dict.get(0, 1.0)
      self._label_weights = torch.tensor([pos_weight], device=self.device)
    else:
      self._label_weights = torch.zeros(len(self.unique_labels))
      for label, weight in class_weights_dict.items():
        self._label_weights[label] = weight
      self._label_weights = self._label_weights.to(self.device)

  def _prepare_dataloader(self, X: np.ndarray, y: np.ndarray = None):
    """
    Prepare a PyTorch DataLoader for the given data.

    Args:
        X: Input images
        y: Labels (if None, dummy labels will be created)

    Returns:
        DataLoader for the data
    """
    # Create tensors for inputs and labels
    if y is None:
      # For prediction, create dummy labels
      y = np.zeros(len(X))

    if self.binary_classification:
      y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(1)
    else:
      y_tensor = torch.tensor(y, dtype=torch.long)

    # Process images and create dataset
    dataset = TensorDataset(
      self._image_processor(X, return_tensors="pt").pixel_values,
      y_tensor,
    )

    # Determine batch size
    if self.batch_size == "auto":
      batch_size = min(32, len(X))
    else:
      batch_size = self.batch_size

    # Set up random generator if needed
    generator = None
    if self.random_state is not None and self.shuffle:
      generator = torch.Generator()
      generator.manual_seed(self.random_state)

    # Create and return the DataLoader
    return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=self.shuffle,
      generator=generator if self.shuffle else None,
    )

  def _prepare_loss_function(self):
    """
    Set up the loss function based on the classification type.
    """
    if self.binary_classification:
      self._criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=self._label_weights if self._label_weights is not None else None
      )
    else:
      self._criterion = torch.nn.CrossEntropyLoss(
        weight=self._label_weights if self._label_weights is not None else None
      )

  def _validate_and_prepare_labels(self, y_train: np.ndarray) -> np.ndarray:
    """
    Validate and prepare the labels for training.

    Args:
        y_train: Training labels

    Returns:
        Processed training labels
    """
    self.unique_labels = np.unique(y_train)

    # Verify binary classification
    if self.binary_classification:
      if len(self.unique_labels) > 2:
        raise ValueError(
          f"Binary classification requires 2 classes, but got {len(self.unique_labels)} classes."
        )
      elif len(self.unique_labels) == 1:
        raise ValueError(
          f"Training data contains only one class: {self.unique_labels[0]}. Need samples from both classes for binary classification."
        )

      # Remap labels to 0 and 1 if needed
      if not np.array_equal(np.sort(self.unique_labels), np.array([0, 1])):
        y_map = {label: i for i, label in enumerate(self.unique_labels)}
        y_train = np.array([y_map[y] for y in y_train])
        if self.verbose:
          print(f"Remapped labels from {self.unique_labels} to [0, 1]")
        self.unique_labels = np.array([0, 1])

    return y_train

  def _prepare_train_validation_split(self, X_train, y_train):
    """
    Prepare training and validation data loaders.

    Args:
        X_train: Training data
        y_train: Training labels

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if self.early_stopping or self.lr_scheduler == "reduce_on_plateau":
      # Split data for validation
      X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=self.validation_fraction,
        random_state=self.random_state,
        stratify=y_train if self.class_weight is not None else None,
      )
      train_dataloader = self._prepare_dataloader(X_train_split, y_train_split)
      val_dataloader = self._prepare_dataloader(X_val, y_val)
      return train_dataloader, val_dataloader
    else:
      # No validation split needed
      train_dataloader = self._prepare_dataloader(X_train, y_train)
      return train_dataloader, None

  def fit(self, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit the model to the training data.

    Args:
        X_train: Training images
        y_train: Training labels

    Returns:
        self
    """
    # Validate and prepare labels
    y_train = self._validate_and_prepare_labels(y_train)

    # Set up model, optimizer, scheduler, and weights
    self._prepare_model()
    self._prepare_optimizer()
    self._prepare_scheduler()
    self._prepare_weights(y_train)

    # Prepare data loaders
    train_dataloader, val_dataloader = self._prepare_train_validation_split(
      X_train, y_train
    )

    # Report class weights if verbose
    if self._label_weights is not None and self.verbose:
      print(f"Using class weights: {self._label_weights}")

    # Set up loss function
    self._prepare_loss_function()

    # Train the model
    self._train_model(train_dataloader, val_dataloader)

    return self

  def _train_model(self, train_dataloader, val_dataloader=None):
    """
    Train the model with the given data loaders.

    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
    """
    best_loss = float("inf")
    no_improvement_count = 0
    self._best_model_state = None

    for epoch in tqdm(range(self.max_iter), desc="Epoch", position=0):
      # Train for one epoch
      avg_train_loss = self._train_epoch(train_dataloader)

      # Evaluate on validation set if available
      if val_dataloader is not None:
        avg_val_loss = self._evaluate_model(val_dataloader)

        # Update learning rate scheduler if needed
        if self.lr_scheduler == "reduce_on_plateau":
          self._scheduler.step(avg_val_loss)

        # Check for early stopping
        if self.early_stopping:
          if avg_val_loss < best_loss - self.tol:
            best_loss = avg_val_loss
            no_improvement_count = 0
            self._best_model_state = self._model.state_dict().copy()
          else:
            no_improvement_count += 1

          if no_improvement_count >= self.n_iter_no_change:
            if self.verbose:
              print(f"Early stopping at epoch {epoch + 1}")
            if self._best_model_state is not None:
              self._model.load_state_dict(self._best_model_state)
            break
        elif avg_val_loss < best_loss:
          best_loss = avg_val_loss
          self._best_model_state = self._model.state_dict().copy()

      # Update schedulers that don't need validation loss
      if self.lr_scheduler in ["cosine_annealing", "step", "exponential"]:
        self._scheduler.step()
        if self.verbose:
          current_lr = self._optimizer.param_groups[0]["lr"]
          print(f"Epoch {epoch + 1} - Current learning rate: {current_lr:.2e}")

      # Print progress
      if self.verbose:
        if val_dataloader is not None:
          print(
            f"Epoch {epoch + 1}/{self.max_iter} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
          )
        else:
          print(f"Epoch {epoch + 1}/{self.max_iter} - Train Loss: {avg_train_loss:.4f}")

    # Save the best model state if none was saved yet
    if self._best_model_state is None:
      self._best_model_state = self._model.state_dict().copy()
    # Load the best model if validation was used
    elif self.early_stopping or val_dataloader is not None:
      self._model.load_state_dict(self._best_model_state)

  def _train_epoch(self, train_dataloader):
    """
    Train for one epoch.

    Args:
        train_dataloader: DataLoader for training data

    Returns:
        Average training loss for the epoch
    """
    self._model.train()
    train_loss = 0.0
    n_batches = 0

    for batch in tqdm(train_dataloader, desc="Batch", position=1, leave=False):
      inputs, labels = batch
      inputs = inputs.to(self.device)
      labels = labels.to(self.device)

      batch_loss = self._train_batch(inputs, labels)
      train_loss += batch_loss
      n_batches += 1

    return train_loss / n_batches

  def _train_batch(self, inputs, labels):
    """
    Train on a single batch.

    Args:
        inputs: Batch of input images
        labels: Batch of labels

    Returns:
        Loss for this batch
    """
    if self.solver == "lbfgs":

      def closure():
        self._optimizer.zero_grad()
        outputs = self._model(inputs, labels=labels)

        if self._label_weights is None:
          loss = outputs.loss
        else:
          logits = outputs.logits
          loss = self._criterion(logits, labels)

        loss.backward()
        return loss

      loss = self._optimizer.step(closure)
      return loss.item()
    else:
      self._optimizer.zero_grad()
      outputs = self._model(inputs, labels=labels)

      if self._label_weights is None:
        loss = outputs.loss
      else:
        logits = outputs.logits
        loss = self._criterion(logits, labels)

      loss.backward()
      self._optimizer.step()

      return loss.item()

  def _evaluate_model(self, val_dataloader):
    """
    Evaluate the model on the validation data.

    Args:
        val_dataloader: DataLoader for validation data

    Returns:
        Average validation loss
    """
    self._model.eval()
    val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():
      for batch in val_dataloader:
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs = self._model(inputs, labels=labels)

        if self._label_weights is None:
          loss = outputs.loss
        else:
          logits = outputs.logits
          loss = self._criterion(logits, labels)

        val_loss += loss.item()
        n_val_batches += 1

    return val_loss / n_val_batches

  def predict(self, X_test: np.ndarray) -> np.ndarray:
    """
    Predict class labels for the input samples.

    Args:
        X_test: Test images

    Returns:
        Predicted class labels
    """
    if self._model is None:
      raise ValueError("Model not trained. Call fit() first.")

    dataloader = self._prepare_dataloader(X_test)

    self._model.eval()

    y_pred = []
    with torch.no_grad():
      for batch in dataloader:
        X_batch = batch[0]
        X_batch = X_batch.to(self.device)

        y_batch_logits = self._model(X_batch).logits
        if self.binary_classification:
          y_batch_pred = (y_batch_logits > 0).float().cpu().numpy()
        else:
          y_batch_pred = torch.max(y_batch_logits, dim=1)[1].cpu().numpy()
        y_pred.append(y_batch_pred)

    y_pred = np.concatenate(y_pred)
    if self.binary_classification:
      y_pred = y_pred.reshape(-1)

    return y_pred

  def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
    """
    Predict class probabilities for the input samples.

    Args:
        X_test: Test images

    Returns:
        Predicted class probabilities
    """
    if self._model is None:
      raise ValueError("Model not trained. Call fit() first.")

    dataloader = self._prepare_dataloader(X_test)

    self._model.eval()

    y_pred_prob = []
    with torch.no_grad():
      for batch in dataloader:
        X_batch = batch[0]
        X_batch = X_batch.to(self.device)

        y_batch_logits = self._model(X_batch).logits
        if self.binary_classification:
          probs = torch.sigmoid(y_batch_logits)
          y_batch_probs = torch.cat([1 - probs, probs], dim=1).cpu().numpy()
        else:
          y_batch_probs = (
            torch.nn.functional.softmax(y_batch_logits, dim=1).cpu().numpy()
          )
        y_pred_prob.append(y_batch_probs)

    return np.concatenate(y_pred_prob)


class ConvNeXTPretrainedClassifier(HGPretrainedClassifier):
  """
  A classifier based on ConvNeXT pre-trained models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str = "facebook/convnextv2-atto-1k-224",
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True,
    rho: float = 0.99,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      binary_classification=binary_classification,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
      class_weight=class_weight,
      solver=solver,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      learning_rate=learning_rate,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      lr_scheduler=lr_scheduler,
      lr_scheduler_patience=lr_scheduler_patience,
      lr_scheduler_factor=lr_scheduler_factor,
      lr_scheduler_min_lr=lr_scheduler_min_lr,
      lr_scheduler_t_max=lr_scheduler_t_max,
      lr_scheduler_step_size=lr_scheduler_step_size,
      verbose=verbose,
    )


class EfficientNetPretrainedClassifier(HGPretrainedClassifier):
  """
  A classifier based on EfficientNet pre-trained models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str = "google/efficientnet-b0",
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "rmsprop",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True,
    rho: float = 0.9,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      binary_classification=binary_classification,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
      class_weight=class_weight,
      solver=solver,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      learning_rate=learning_rate,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      lr_scheduler=lr_scheduler,
      lr_scheduler_patience=lr_scheduler_patience,
      lr_scheduler_factor=lr_scheduler_factor,
      lr_scheduler_min_lr=lr_scheduler_min_lr,
      lr_scheduler_t_max=lr_scheduler_t_max,
      lr_scheduler_step_size=lr_scheduler_step_size,
      verbose=verbose,
    )


class ResNetPretrainedClassifier(HGPretrainedClassifier):
  """
  A classifier based on ResNet pre-trained models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str = "microsoft/resnet-18",
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "sgd",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.875,
    nesterovs_momentum: bool = True,
    rho: float = 0.99,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      binary_classification=binary_classification,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
      class_weight=class_weight,
      solver=solver,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      learning_rate=learning_rate,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      lr_scheduler=lr_scheduler,
      lr_scheduler_patience=lr_scheduler_patience,
      lr_scheduler_factor=lr_scheduler_factor,
      lr_scheduler_min_lr=lr_scheduler_min_lr,
      lr_scheduler_t_max=lr_scheduler_t_max,
      lr_scheduler_step_size=lr_scheduler_step_size,
      verbose=verbose,
    )


class SwinPretrainedClassifier(HGPretrainedClassifier):
  """
  A classifier based on Swin Transformer pre-trained models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str = "microsoft/swin-tiny-patch4-window7-224",
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True,
    rho: float = 0.99,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      binary_classification=binary_classification,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
      class_weight=class_weight,
      solver=solver,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      learning_rate=learning_rate,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      lr_scheduler=lr_scheduler,
      lr_scheduler_patience=lr_scheduler_patience,
      lr_scheduler_factor=lr_scheduler_factor,
      lr_scheduler_min_lr=lr_scheduler_min_lr,
      lr_scheduler_t_max=lr_scheduler_t_max,
      lr_scheduler_step_size=lr_scheduler_step_size,
      verbose=verbose,
    )


class ViTPretrainedClassifier(HGPretrainedClassifier):
  """
  A classifier based on Vision Transformer (ViT) pre-trained models.
  """

  def __init__(
    self,
    pretrained_model_name_or_path: str = "google/vit-base-patch16-224-in21k",
    binary_classification: bool = False,
    freeze_pretrained: bool = False,
    freeze_except_layers: list = None,
    class_weight: Union[Dict[int, float], str, None] = None,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "sgd",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,
    tol: float = 1e-4,
    n_iter_no_change=10,
    learning_rate: float = 1e-5,
    alpha: float = 0.0001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    momentum: float = 0.9,
    nesterovs_momentum: bool = True,
    rho: float = 0.99,
    max_fun: int = 15000,
    lr_scheduler: Literal[
      "reduce_on_plateau", "cosine_annealing", "step", "exponential", None
    ] = None,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.1,
    lr_scheduler_min_lr: Union[List[float], float] = 0,
    lr_scheduler_t_max: int = None,
    lr_scheduler_step_size: int = 10,
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      binary_classification=binary_classification,
      freeze_pretrained=freeze_pretrained,
      freeze_except_layers=freeze_except_layers,
      class_weight=class_weight,
      solver=solver,
      random_state=random_state,
      shuffle=shuffle,
      batch_size=batch_size,
      max_iter=max_iter,
      early_stopping=early_stopping,
      validation_fraction=validation_fraction,
      tol=tol,
      n_iter_no_change=n_iter_no_change,
      learning_rate=learning_rate,
      alpha=alpha,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
      momentum=momentum,
      nesterovs_momentum=nesterovs_momentum,
      rho=rho,
      max_fun=max_fun,
      lr_scheduler=lr_scheduler,
      lr_scheduler_patience=lr_scheduler_patience,
      lr_scheduler_factor=lr_scheduler_factor,
      lr_scheduler_min_lr=lr_scheduler_min_lr,
      lr_scheduler_t_max=lr_scheduler_t_max,
      lr_scheduler_step_size=lr_scheduler_step_size,
      verbose=verbose,
    )
