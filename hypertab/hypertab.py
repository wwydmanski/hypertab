from .hypernetwork import Hypernetwork
from sklearn.base import BaseEstimator, ClassifierMixin
from .interfaces import HypernetworkSklearnInterface
import numpy as np
import torch

class HyperTabClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        subsample: float = 0.5,
        device: str = "cpu",
        hidden_dims: int = 10,
        test_nodes: int = 100,
        verbose: bool = False,
        epochs: int = 50,
        lr: float = 3e-4,
        batch_size: int = 128,
    ) -> None:
        """Initialize HyperTab classifier.

        Args:
            subsample (float): ratio for subsampling the columns of the input matrix
            device (str): device to use for training
            hidden_dims (int): number of neurons in the hidden layer of the target network
            test_nodes (int): number of test nodes to use. Smaller values will result in faster training,
                but less accurate results. This number should be inverse proportional
                to the number of samples in the dataset.
            verbose (bool): whether to print training progress
            epochs (int): number of epochs to train for
            lr (float): learning rate
            batch_size (int): batch size to use for training
        """

        self.subsample: float = subsample
        self.device: str = device
        self.hidden_dims: int = hidden_dims
        self.test_nodes: int = test_nodes
        self.verbose: bool = verbose
        self.epochs: int = epochs
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.interface = None

    def fit(self, X, y):
        """ Train the model on the given data.

        Args:
            X (np.ndarray): input data
            y (np.ndarray): target data
        """

        try:
            X = X.numpy()
            y = y.numpy()
        except AttributeError:
            pass

        n_unique = len(np.unique(y))
        input_dims = X.shape[1]
        target_dim = int(input_dims * self.subsample)
        target_architecture = [
            (target_dim, self.hidden_dims),
            (self.hidden_dims, n_unique),
        ]

        self.hypernet = Hypernetwork(
            input_dims=input_dims,
            target_architecture=target_architecture,
            test_nodes=self.test_nodes,
            device=self.device,
        )
        self.hypernet.to(self.device)
        interface = HypernetworkSklearnInterface(
            self.hypernet,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            verbose=self.verbose,
        )

        # cast X, y to torch
        X = torch.from_numpy(X).to(torch.float32)
        y = torch.from_numpy(y).to(torch.long)

        interface.fit(X, y)
        self.interface = interface
    
    def predict(self, X):
        """ Return the predicted value of each sample in X """
        try:
            X = X.numpy()
        except AttributeError:
            pass

        if self.interface is None:
            raise ValueError("Model not trained yet")
        
        X = torch.from_numpy(X).to(torch.float32)
        return self.interface.predict(X)

    def predict_proba(self, X):
        """ Return the predicted class probability of each sample in X """
        try:
            X = X.numpy()
        except AttributeError:
            pass

        if self.interface is None:
            raise ValueError("Model not trained yet")

        X = torch.from_numpy(X).to(torch.float32)
        return self.interface.predict_proba(X)

