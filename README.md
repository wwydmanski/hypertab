<a target="_blank" href="https://colab.research.google.com/github/wwydmanski/hypertab/blob/master/notebooks/HyperTab_ablation_study.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# HyperTab

HyperTab is a hypernetwork-based classifier for small tabular datasets. It's especially efficient when the number of samples is smaller than 500. The larger the dataset, the smaller is the advantage of using HyperTab over other methods.

## Installation

```bash
pip install hypertab
```

## Usage

```python
from hypertab import HyperTabClassifier
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

clf = HyperTabClassifier(0.5, device=DEVICE, test_nodes=100, epochs=10, hidden_dims=5)
clf.fit(X, y)
clf.predict(X)
```

## Performance
| **Dataset**                   | **XGBoost** | **DN**     | **RF**               | **HyperTab**         | **Node**             |
|---------------------------------|---------------|--------------|------------------------|------------------------|------------------------|
| **Wisconsin Breast Cancer**   | 93.85 | 95.58 | 95.96 | **97.58** | 96.19 |
| **Connectionist Bench**       | 83.52 | 79.02 | 83.50 | **87.09** | 85.61 |
| **Dermatology**               | 96.05 | 97.80 | 97.21 | 97.82 | **97.99** |
| **Glass**                     | 94.74 | 46.96 | 97.02 | **98.36** | 44.90 |
| **Promoter**                  | 81.88 | 78.91 | 85.94 | **89.06** | 83.75 |
| **Ionosphere**                | 90.67 | 93.43 | 92.43 | **94.52** | 91.03 |
| **Libras**                    | 74.38 | 81.54 | 77.42 | **85.22** | 82.72 |
| **Lymphography**              | 85.94 | 85.74 | **87.19** | 83.90 | 83.93 |
| **Parkinsons**                | 86.35 | 74.96 | 86.84 | **95.27** | 80.20 |
| **Zoo**                       | 92.86 | 72.62 | 92.62 | **95.27** | 89.05 |
| **Hill-Valley without noise** | 65.53 | 56.39 | 57.33 | **70.59** | 52.71 |
| **Hill-Valley with noise**    | 58.45 | 56.06 | 55.66 | **67.56** | 51.09 |
| **OpenML 1086**               | 60.61 | 33.33 | 51.24 | **76.60** | 68.39 |
| **Heart**                     | 79.17 | 82.62 | 81.10 | **83.33** | 82.38 |
| **Mean rank**                 | 3.50  | 3.78  | 3.07  | 1.35      | 3.29  |
