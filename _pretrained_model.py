from typing import Literal
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import numpy as np
from torch.optim import AdamW, Adam, SGD, RMSprop, LBFGS
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModelForImageClassification


class HGPretrainedClassifier(BaseEstimator, ClassifierMixin):
  def __init__(
    self,
    pretrained_model_name_or_path: str,
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    self.pretrained_model_name_or_path = pretrained_model_name_or_path
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
    self.verbose = verbose

    if self.random_state is not None:
      self.rng = np.random.RandomState(self.random_state)
    else:
      self.rng = np.random.RandomState()

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      torch.cuda.manual_seed(self.random_state)
      torch.cuda.manual_seed_all(self.random_state)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    else:
      self.device = torch.device("cpu")

  def _prepare_model(self):
    self._image_processor = AutoImageProcessor.from_pretrained(
      self.pretrained_model_name_or_path, use_fast=True
    )
    self._model = AutoModelForImageClassification.from_pretrained(
      self.pretrained_model_name_or_path
    )
    self._model.to(self.device)

  def _prepare_optimizer(self):
    if self.solver == "adamw":
      self._optimizer = AdamW(
        self._model.parameters(),
        lr=self.learning_rate,
        betas=(self.beta_1, self.beta_2),
        eps=self.epsilon,
        weight_decay=self.alpha,
      )
    elif self.solver == "adam":
      self._optimizer = Adam(
        self._model.parameters(),
        lr=self.learning_rate,
        betas=(self.beta_1, self.beta_2),
        eps=self.epsilon,
        weight_decay=self.alpha,
      )
    elif self.solver == "sgd":
      self._optimizer = SGD(
        self._model.parameters(),
        lr=self.learning_rate,
        momentum=self.momentum,
        weight_decay=self.alpha,
        nesterov=self.nesterovs_momentum,
      )
    elif self.solver == "rmsprop":
      self._optimizer = RMSprop(
        self._model.parameters(),
        lr=self.learning_rate,
        eps=self.epsilon,
        weight_decay=self.alpha,
        momentum=self.momentum,
        alpha=self.rho,
      )
    elif self.solver == "lbfgs":
      self._optimizer = LBFGS(
        self._model.parameters(),
        lr=self.learning_rate,
        max_iter=self.max_iter,
        max_eval=self.max_fun,
      )

  def _prepare_dataloader(self, X: np.ndarray, y: np.ndarray):
    dataset = TensorDataset(
      self._image_processor(X, return_tensors="pt").pixel_values,
      torch.tensor(y, dtype=torch.long),
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
      dataset,
      batch_size=batch_size,
      shuffle=self.shuffle,
      generator=generator if self.shuffle else None,
    )

  def fit(self, X_train: np.ndarray, y_train: np.ndarray):
    self._prepare_model()
    self._prepare_optimizer()

    if self.early_stopping:
      X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=self.validation_fraction,
        random_state=self.random_state,
      )
      train_dataloader = self._prepare_dataloader(X_train_split, y_train_split)
      val_dataloader = self._prepare_dataloader(X_val, y_val)
    else:
      train_dataloader = self._prepare_dataloader(X_train, y_train)

    best_loss = float("inf")
    no_improvement_count = 0
    for epoch in range(self.max_iter):
      # Train
      self._model.train()
      train_loss = 0.0
      n_batches = 0

      for batch in train_dataloader:
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if self.solver == "lbfgs":

          def closure():
            self._optimizer.zero_grad()
            outputs = self._model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            return loss

          loss = self._optimizer.step(closure)
        else:
          self._optimizer.zero_grad()
          outputs = self._model(inputs, labels=labels)
          loss = outputs.loss
          loss.backward()
          self._optimizer.step()

        train_loss += loss.item()
        n_batches += 1

      avg_train_loss = train_loss / n_batches

      # Validation for early stopping
      if self.early_stopping:
        self._model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
          for batch in val_dataloader:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self._model(inputs, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()
            n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches

        # Check for improvement
        if avg_val_loss < best_loss - self.tol:
          best_loss = avg_val_loss
          no_improvement_count = 0
          self._best_model_state = self._model.state_dict().copy()
        else:
          no_improvement_count += 1

        # Check for early stop
        if no_improvement_count >= self.n_iter_no_change:
          if self.verbose:
            print(f"Early stopping at epoch {epoch + 1}")
          self._model.load_state_dict(self._best_model_state)
          break

      if self.verbose:
        if self.early_stopping:
          print(
            f"Epoch {epoch + 1}/{self.max_iter} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
          )
        else:
          print(f"Epoch {epoch + 1}/{self.max_iter} - Train Loss: {avg_train_loss:.4f}")

    # Save the final model if early stopping wasn't used
    if not self.early_stopping:
      self._best_model_state = self._model.state_dict().copy()

    return self

  def predict(self, X_test: np.ndarray) -> np.ndarray:
    if self._model is None:
      raise ValueError("Model not trained. Call fit() first.")

    dataloader = self._prepare_dataloader(X_test, np.zeros(len(X_test)))

    self._model.eval()

    y_pred = []
    with torch.no_grad():
      for batch in dataloader:
        X_batch = batch[0]
        X_batch = X_batch.to(self.device)

        y_batch_logits = self._model(X_batch).logits
        y_batch_pred = torch.max(y_batch_logits, dim=1)[1]
        y_pred.append(y_batch_pred.cpu().numpy())

    return np.concatenate(y_pred)

  def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
    if self._model is None:
      raise ValueError("Model not trained. Call fit() first.")

    dataloader = self._prepare_dataloader(X_test, np.zeros(len(X_test)))

    self._model.eval()

    y_pred_prob = []
    with torch.no_grad():
      for batch in dataloader:
        X_batch = batch[0]
        X_batch = X_batch.to(self.device)

        y_batch_logits = self._model(X_batch).logits
        y_batch_probs = torch.nn.functional.softmax(y_batch_logits, dim=1)
        y_pred_prob.append(y_batch_probs.cpu().numpy())

    return np.concatenate(y_pred_prob)


class ConvNeXTPretrainedClassifier(HGPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "facebook/convnextv2-tiny-1k-224",
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
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
      verbose=verbose,
    )


class EfficientNetPretrainedClassifier(HGPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "google/efficientnet-b7",
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "rmsprop",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.9,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
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
      verbose=verbose,
    )


class ResNetPretrainedClassifier(HGPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "microsoft/resnet-50",
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "sgd",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.875,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
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
      verbose=verbose,
    )


class SwinPretrainedClassifier(HGPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "microsoft/swinv2-tiny-patch4-window8-256",
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "adamw",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
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
      verbose=verbose,
    )


class ViTPretrainedClassifier(HGPretrainedClassifier):
  def __init__(
    self,
    pretrained_model_name_or_path: str = "google/vit-base-patch16-224",
    solver: Literal["adamw", "adam", "sgd", "rmsprop", "lbfgs"] = "sgd",
    random_state: int = None,
    shuffle: bool = True,
    batch_size: int | str = "auto",
    max_iter: int = 200,
    early_stopping: bool = False,
    validation_fraction=0.1,  # Only used if early_stopping is True.
    tol: float = 1e-4,  # Only used if early_stopping is True.
    n_iter_no_change=10,  # Only used if early_stopping is True.
    learning_rate: float = 1e-3,
    alpha: float = 0.0001,
    beta_1: float = 0.9,  # Only used when solver='adam' or 'adamw'.
    beta_2: float = 0.999,  # Only used when solver='adam' or 'adamw'.
    epsilon: float = 1e-8,  # Only used when solver='adam' or 'adamw'.
    momentum: float = 0.9,  # Only used when solver='sgd'.
    nesterovs_momentum: bool = True,  # Only used when solver='sgd', and momentum greater than 0.
    rho: float = 0.99,  # Only used when solver='rmsprop'.
    max_fun: int = 15000,  # Only used when solver='lbfgs'.
    verbose: bool = False,
  ):
    super().__init__(
      pretrained_model_name_or_path=pretrained_model_name_or_path,
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
      verbose=verbose,
    )
