import sys
sys.path.append("..")

from hypertab import HyperTabClassifier
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def test_basic_prediction():
    # Set the random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a dataset
    X = np.arange(10240).reshape(512, -1)
    y = X[:, 0] > 500

    # Create a classifier
    clf = HyperTabClassifier(epochs=40, device="cuda:0")

    # Train the classifier
    clf.fit(X, y)

    # Predict the labels
    y_pred = clf.predict(X)

    assert y_pred.shape == (512,)
    assert accuracy_score(y, y_pred) > 0.9
        
if __name__ == "__main__":
    test_basic_prediction()