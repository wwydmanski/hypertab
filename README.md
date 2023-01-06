<a target="_blank" href="https://colab.research.google.com/github/wwydmanski/hypertab/blob/master/notebooks/HyperTab_ablation_study.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

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

## Performance
| **Dataset**                   | **XGBoost** | **DN**     | **RF**               | **HyperTab**         | **Node**             |
|---------------------------------|---------------|--------------|------------------------|------------------------|------------------------|
| **Wisconsin Breast Cancer**   | 93.85 (1.44)  | 95.58 (1.04) | 95.96 (1.52)           | **97.58 (1.11)** | 96.19 (1.11)           |
| **Connectionist Bench**       | 83.52 (3.94)  | 79.02 (5.29) | 83.50 (5.55)           | **87.09 (5.53)** | 85.61 (3.48)           |
| **Dermatology**               | 96.05 (0.89)  | 97.80 (1.17) | 97.21 (1.66)           | 97.82 (1.24)           | **97.99 (1.20)** |
| **Glass**                     | 94.74 (3.91)  | 46.96 (2.56) | 97.02 (1.51)           | **98.36 (3.21)** | 44.90 (1.90)           |
| **Promoter**                  | 81.88 (5.59)  | 78.91 (3.93) | 85.94 (6.79)           | **89.06 (5.41)** | 83.75 (4.64)           |
| **Ionosphere**                | 90.67 (2.75)  | 93.43 (3.72) | 92.43 (2.60)           | **94.52 (1.47)** | 91.03 (1.79)           |
| **Libras**                    | 74.38 (4.55)  | 81.54 (3.99) | 77.42 (3.88)           | **85.22 (2.92)** | 82.72 (3.27)           |
| **Lymphography**              | 85.94 (3.14)  | 85.74 (5.28) | **87.19 (4.33)** | 83.90 (5.01)           | 83.93 (5.82)           |
| **Parkinsons**                | 86.35 (4.77)  | 74.96 (4.90) | 86.84 (6.26)           | **95.27 (3.06)** | 80.20 (5.29)           |
| **Zoo**                       | 92.86 (8.75)  | 72.62 (4.96) | 92.62 (7.97)           | **95.27 (3.06)** | 89.05 (3.98)           |
| **Hill-Valley without noise** | 65.53 (0.00)  | 56.39 (2.89) | 57.33 (0.00)           | **70.59 (4.90)** | 52.71 (0.34)           |
| **Hill-Valley with noise**    | 58.45 (0.00)  | 56.06 (1.65) | 55.66 (0.00)           | **67.56 (8.17)** | 51.09 (0.26)           |
| **OpenML 1086**               | 60.61 (8.80)  | ()           | 51.24 (7.53)           | **76.60 (4.48)** | 68.39 (10.82)          |
| **Mean rank**                 | 3.38 (1.26)   | 3.83 (1.03)  | 3.00 (1.00)            | 1.38 (1.12)            | 3.31 (1.38)            |
