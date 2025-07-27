
# Entropy Estimators

A Python library for computing entropy and divergence estimates from discrete data using various statistical estimators. Includes advanced estimators like James-Stein shrinkage, Chao-Shen, and Nemenman-Shafee-Bialek (NSB), as well as divergence metrics like KL and Jensen-Shannon.

---

## 📦 Installation

Install locally in development mode:

```bash
git clone https://github.com/fermosc24/entropy-estimators.git
cd entropy-estimators
pip install -e .
```

Dependencies:
- `numpy`
- `scipy`
- `mpmath`

---

## 🧠 Features

- **Entropy estimation methods:**
  - Maximum Likelihood (MLE)
  - James-Stein Shrinkage Estimator (JSE)
  - Chao-Shen (CAE)
  - Chao-Wang-Jost (CWJ)
  - Nemenman-Bialek-de Ruyter van Steveninck (NBRS)
  - Nemenman-Shafee-Bialek (NSB)

- **Divergence metrics:**
  - KL Divergence (James-Stein smoothed)
  - Jensen-Shannon Divergence

- **Utility tools:**
  - `FreqShrink`: James-Stein frequency smoothing
  - `sample_frequencies`: Sampling from discrete frequency arrays
  - `dict_to_ndarray`: Convert labeled dictionary to tensor + label maps

---

## 🔧 Usage

### Entropy estimation

```python
from entropy_estimators import Entropy

counts = [5, 3, 2, 0, 1]
entropy = Entropy(counts, method="NSB")
print("Entropy:", entropy)
```

### Jensen-Shannon divergence

```python
from entropy_estimators import JS_JensenShannon

a = [10, 0, 5, 2]
b = [5, 3, 5, 4]

jsd = JS_JensenShannon(a, b)
print("Jensen-Shannon divergence:", jsd)
```

---

## 📁 Project Structure

```
entropy-estimators/
├── setup.py
├── pyproject.toml
├── src/
│   └── entropy_estimators/
│       ├── __init__.py
│       └── core.py
├── tests/
│   └── test_entropy_estimators.py
```

---

## 📜 License

MIT License © 2025 [fermosc24](https://github.com/fermosc24)
