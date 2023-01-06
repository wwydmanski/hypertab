# HyperTab

HyperTab is a hypernetwork-based classifier for small tabular datasets. 

## Installation

```bash
pip install hypertab
```

## Usage

```python
from hypertab import HyperTabClassifier
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

clf = HyperTabClassifier(0.2, device=DEVICE, test_nodes=100, epochs=10, hidden_dims=5)
clf.fit(X, y)
clf.predict(X)
```