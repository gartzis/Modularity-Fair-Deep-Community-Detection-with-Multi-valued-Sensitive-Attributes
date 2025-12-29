# Modularity-Fair Deep Community Detection (ICDM’25 + KAIS extension)

This repository provides implementations of modularity-based community detection methods that incorporate **connectivity fairness** with respect to a sensitive attribute.
It includes the **binary** setting (two groups) and the **multivalued** setting (multiple groups), as presented in:

- **ICDM’25 (DM624):** *Modularity-Fair Deep Community Detection*
- **KAIS journal extension:** *Modularity-Fair Deep Community Detection with Multi-valued Sensitive Attributes*





## Directory Structure

- `main_spectral.py` – Main script for running spectral clustering variants.
- `main_dmon.py` – Main script for running DMoN-based clustering with modified adjacency matrices.
- `main_deep.py` – Main script for running deep clustering with fairness-aware loss functions.
- `main_multi_group_deep.py` – Main script for running multi-group deep clustering variants.

- `Community Detection/`
  - `spectralClustering.py`: Implements **GroupSpectral** and **DiversitySpectral** based on the modified modularity matrices.
  - `dmonClustering.py`: Implements **GroupDMoN** and **DiversityDMoN** using modified adjacency matrices.
  - `deepClustering.py`: Implements **DeepFairness**, **DeepGroup**, and **DeepDiversity** by customizing the DMoN loss function.
  - `multi_deepClustering.py`: Implements **MultiDeepGroup**, **MultiDeepFairness**, and **MultiDeepDiversity** for multivalued sensitive attributes.

- `Community Detection/tools/`
  - `dmon.py`: Implementation of the DMoN layer, extended with fairness and diversity-aware loss terms.
  - `gcn.py`, `utils.py`, `metrics.py`: Additional model utilities.

- `Algorithms/` – Metric computations and helper routines.
- `Data/` – Synthetic datasets (binary and multivalued).
- `Synth Results/` – Output folders created by the scripts.


---

## Algorithms

### Input-based Spectral Algorithms

- **GroupSpectral**: Enhances group-modularity using a modified modularity matrix $B(\lambda) = (1-\lambda)B + \lambda B_R$.
- **DiversitySpectral**: Promotes diversity using a modified modularity matrix $B(\lambda) = (1-\lambda)B + \lambda B_{div}$.

### Input-based DMoN Algorithms

- **GroupDMoN**: Modifies the adjacency matrix to emphasize edges involving a target group $A(\lambda) = (1-\lambda)A + \lambda A_R$.
- **DiversityDMoN**: Emphasizes inter-group edges to promote diversity within clusters $A(\lambda) = (1-\lambda)A + \lambda A_{div}$.

### Loss-based Deep Clustering (Binary)

- **DeepFairness**: Minimizes the absolute difference between red and blue group modularity.
- **DeepGroup**: Maximizes modularity for a protected group in addition to overall modularity.
- **DeepDiversity**: Maximizes diversity.

### Loss-based Deep Clustering (Multivalued)

- **DeepDiversity**: Maximizes diversity.
- **MultiDeepGroup**: Maximizes the minimum-group modularity.
- **MultiDeepFairness**: Minimizes the max--min group modularity gap.
- **MultiDeepDiversity**: Maximizes the minimum group-aware diversity.


## Requirements

The code is research-oriented (not packaged). A typical environment works with:

- Python 3.9+
- `numpy`, `scipy`, `pandas`, `networkx`
- `scikit-learn`, `matplotlib`
- `tensorflow`

Example setup:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install --upgrade pip
pip install numpy scipy pandas networkx scikit-learn matplotlib tensorflow
```

## Running the Code

All methods expect two input files:

- `your_graph.edgelist`: Edge list with space-separated node pairs.
- `your_graph.csv`: CSV with `nodes` and `attribute` columns  

### Example

```bash
python main_spectral.py
python main_dmon.py
python main_deep.py
python main_multi_group_deep.py
```


---

## Input format

Each run uses:

1) **Edge list** (`.edgelist`)  
One edge per line:
```
u v
u w
...
```

2) **Node attributes** (`.csv`)  
Must contain:
- `nodes` : node id (must match the edgelist ids)
- `attribute` : sensitive group id (binary or multivalued)

Example:
```csv
nodes,attribute
0,2
1,0
2,1
...
```




## Outputs

Depending on the script/method, outputs include:
- community assignments (CSV)
- method-level metrics
- optional plots for the produced partitions

Output paths are created automatically under `Synth Results/`.


---

## Table 1: Summary of the proposed algorithms

| Method | Type | Objective | Attribute |
|---|---|---|---|
| **GroupSpectral** | Input-based, Spectral | Group modularity | Binary |
| **DiversitySpectral** | Input-based, Spectral | Diversity | Binary |
| **GroupDMoN** | Input-based, Deep | Group modularity | Binary |
| **DiversityDMoN** | Input-based, Deep | Diversity | Binary |
| **DeepGroup** | Loss-based, Deep | Group modularity | Binary |
| **DeepDiversity** | Loss-based, Deep | Diversity | Binary, Multivalued |
| **DeepFairness** | Loss-based, Deep | Fairness | Binary |
| **MultiDeepDiversity** | Loss-based, Deep | Group-aware Diversity | Multivalued |
| **MultiDeepFairness** | Loss-based, Deep | Multi-Fairness | Multivalued |
| **MultiDeepGroup** | Loss-based, Deep | Min group Fairness | Multivalued |

**Binary**: two groups.  
**Multivalued**: multiple groups.

---

## Citation

If you use this code, please cite the ICDM’25 paper and/or the KAIS journal extension (multivalued setting), depending on what you use.
