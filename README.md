# Modularity-Fair Deep Community Detection


This repository implements the community detection algorithms described in our ICDM 2025 paper **"Modularity-Fair Deep Community Detection"** , which proposes fairness-aware clustering techniques based on **group modularity**. We introduce three families of algorithms—**spectral**, **DMoN-based**, and **deep clustering with fairness-aware loss**—each designed to balance structural quality with different fairness criteria.

## Directory Structure

- **`main_spectral.py`** – Main script for running spectral clustering variants.
- **`main_dmon.py`** – Main script for running DMoN-based clustering with modified adjacency matrices.
- **`main_deep.py`** – Main script for running deep clustering with fairness-aware loss functions.

- **`Community Detection/`**
  - `spectralClustering.py`: Implements **GroupSpectral** and **DiversitySpectral** based on th modified modularity matrices.
  - `dmonClustering.py`: Implements **GroupDMoN** and **DiversityDMoN** using modified adjacency matrices.
  - `deepClustering.py`: Implements **DeepFairness**, **DeepGroup**, and **DeepDiversity** by customizing the DMoN loss function.

- **`Community Detection/tools/`**
  - `dmon.py`: Implementation of the **DMoN layer**, extended with fairness and diversity-aware loss terms.
  - `gcn.py`, `utils.py`, `metrics.py`: Additional model utilities.

---

## Algorithms

### Input based Spectral Algorithms
- **GroupSpectral**: Enhances group-modularity using a modified modularity matrix `B(λ) = (1−λ)B + λBR`.
- **DiversitySpectral**: Promotes diversity using a modified modularity matrix `B(λ) = (1−λ)B +Bdiv`.

### Input based DMoN Algorithms
- **GroupDMoN**: Modifies the adjacency matrix to emphasize edges involving a target group `A(λ) = (1−λ)A + λAR`.
- **DiversityDMoN**: Emphasizes inter-group edges to promote diversity within clusters `A(λ) = (1−λ)A +Adiv`.

### Loss based Deep Clustering
- **DeepFairness**: Minimizes the absolute difference between red and blue group modularity.
- **DeepGroup**: Maximizes modularity for a protected group in addition to overall modularity.
- **DeepDiversity**: Maximizes diversity.

## Running the Code

All methods expect two input files:
- `your_graph.edgelist`: Edge list with space-separated node pairs.
- `your_graph.csv`: CSV with `nodes` and `attribute` columns (binary attribute: e.g., 0=red, 1=blue).

### Example
```bash
python main_spectral.py
python main_dmon.py
python main_deep.py
```

### Fairness Parameters

- **`λ` (lambda)** controls the trade-off between structural modularity and fairness objectives, taking values from **0** (optimize only modularity) to **1** (optimize only fairness criterion).
- **`ϕ` (phi)** is used in `DeepFairness` to penalize the imbalance between red and blue group modularity, encouraging equal connectivity across groups.


## Summary of Fairness Parameters Usage

| Algorithm         | Uses `λ` | Uses `ϕ` | Purpose                                                                 |
|------------------|----------|----------|-------------------------------------------------------------------------|
| GroupSpectral     | ✅       | ❌       | Balance modularity with group modularity                    |
| DiversitySpectral | ✅       | ❌       | Balance modularity with diversity               |
| GroupDMoN         | ✅       | ❌       | Modify adjacency to emphasize protected group edges                   |
| DiversityDMoN     | ✅       | ❌       | Modify adjacency to emphasize diversity             |
| DeepGroup         | ✅       | ❌       | Loss trade-off between modularity and group modularity                |
| DeepDiversity     | ✅       | ❌       | Loss trade-off between modularity and diversity                       |
| DeepFairness      | ❌       | ✅       | Penalizes modularity imbalance between groups |

<img width="191" height="20" alt="image" src="https://github.com/user-attachments/assets/a1786e79-32f9-4a8e-99e1-986cf54901cc" />
