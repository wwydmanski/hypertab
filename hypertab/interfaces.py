import torch
import numpy as np
from .training_utils import get_dataloader, train_model, basic_train_loop

class SimpleSklearnInterface:
    def __init__(self, network, batch_size=128, epochs=10, lr=3e-4, device="cuda:0", loss=torch.nn.CrossEntropyLoss()):
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.criterion = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def fit(self, X, y):
        try:
            X = torch.from_numpy(X).to(torch.float32)
            y = torch.from_numpy(y).to(torch.long)
        except TypeError:
            pass
        train_data = get_dataloader(X, y, batch_size=self.batch_size)
        basic_train_loop(self.network, self.optimizer, self.criterion, train_data, self.epochs, self.device)

    def predict(self, X):
        res = []
        batch_size = 32
        self.network.eval()
        try:
            X = torch.from_numpy(X).to(torch.float32)
        except TypeError:
            pass

        for i in range(0, len(X), batch_size):
            predictions = (
                self.network(
                    X[i : i + batch_size].to(torch.float32).to(self.device)
                )
                .cpu()
                .detach()
                .numpy()
            )
            if predictions.shape[1] > 1:
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = np.round(predictions).astype(int)
            res.append(predictions)
        return np.concatenate(res)

    def predict_proba(self, X):
        res = []
        batch_size = 32
        self.network.eval()
        
        try:
            X = torch.from_numpy(X).to(torch.float32)
        except TypeError:
            pass

        for i in range(0, len(X), batch_size):
            predictions = (
                self.network(
                    X[i : i + batch_size].to(torch.float32).to(self.device)
                )
                .cpu()
                .detach()
                .numpy()
            )
            res.append(predictions)
        return np.concatenate(res)
    
class HypernetworkSklearnInterface:
    def __init__(self, network, batch_size=128, epochs=10, lr=3e-4, device="cuda:0", verbose=True):
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.batch_callback = None
        self.epoch_callback = None

    def fit(self, X, y):
        try:
            X = torch.from_numpy(X).to(torch.float32)
            y = torch.from_numpy(y).to(torch.long)
        except TypeError:
            pass
        train_data = get_dataloader(X, y, batch_size=self.batch_size)
        train_model(
            self.network,
            self.optimizer,
            self.criterion,
            train_data,
            epochs=self.epochs,
            device=self.network.device,
            batch_callback=self.batch_callback,
            epoch_callback=self.epoch_callback,
            verbose=self.verbose,
        )

    def predict(self, X):
        self.network.eval()
        res = []
        batch_size = 32
        try:
            X = torch.from_numpy(X).to(torch.float32)
        except TypeError:
            pass

        for i in range(0, len(X), batch_size):
            predictions = (
                self.network(
                    X[i : i + batch_size].to(torch.float32).to(self.network.device)
                )
                .cpu()
                .detach()
                .numpy()
            )
            predictions = np.argmax(predictions, axis=1)
            res.append(predictions)
        return np.concatenate(res)

    def predict_proba(self, X):
        self.network.eval()
        
        res = []
        batch_size = 32
        try:
            X = torch.from_numpy(X).to(torch.float32).to(self.network.device)
        except TypeError:
            pass

        for i in range(0, len(X), batch_size):
            predictions = (
                self.network(
                    X[i : i + batch_size].to(torch.float32).to(self.network.device)
                )
                .cpu()
                .detach()
                .numpy()
            )
            res.append(predictions)
        return np.concatenate(res)
