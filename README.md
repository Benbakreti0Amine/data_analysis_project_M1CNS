# Clustering Analysis of the Pumpkin Seeds Dataset

Group project for the data analysis course of the M1 Computer and Network Systems at Université Paris-Saclay.

The goal is to find natural groups in a real-world dataset with unsupervised learning, compare several clustering methods, and check whether dimensionality reduction improves the results.

## Dataset

[Pumpkin Seeds](https://www.muratkoklu.com/datasets/) (Koklu et al., 2021): 2,500 seeds described by 12 morphological features (area, perimeter, axis lengths, roundness, compactness...). Each seed belongs to one of 2 varieties (1,300 and 1,200 samples), which we only use to evaluate the clusters. There are no missing values, and the features are standardized before clustering.

## Methods

1. **Clustering**: K-Means, hierarchical clustering (single, complete and Ward linkage) and spectral clustering (RBF and nearest-neighbors affinity).
2. **Number of clusters**: chosen with the silhouette score for k = 2 to 10.
3. **Evaluation against the real varieties**: contingency matrix, Rand Index, Adjusted Rand Index (ARI), homogeneity, completeness and V-measure.
4. **PCA**: dimensionality reduction, correlation circle and interpretation of the components, then clustering again on the reduced data.
5. **CCA**: canonical correlation analysis between size variables and shape variables.

## Results

The silhouette score selects **2 clusters** for every method, which matches the 2 real varieties.

| Method (k = 2) | ARI | V-measure | Silhouette |
|---|---|---|---|
| K-Means | 0.35 | 0.27 | 0.31 |
| Spectral clustering (nearest neighbors) | 0.35 | 0.27 | 0.29 |
| Hierarchical clustering (Ward) | 0.42 | 0.35 | 0.24 |
| **Hierarchical (Ward) on PCA, 3 components** | **0.56** | **0.45** | **0.30** |

- **Hierarchical clustering with Ward linkage** gives the best agreement with the real varieties on the original features.
- **PCA helps**: the first 3 components keep 91.9% of the variance (PC1: 49.2%, PC2: 34.0%) and remove redundant size features, which raises the ARI from 0.42 to 0.56.
- Spectral clustering with an RBF affinity has a high silhouette score (0.79) but puts almost all seeds in one cluster, so the score is misleading on its own.
- CCA finds a strong link between size and shape variables, but it is less useful than PCA to prepare the data for clustering.

## Files

| File | Content |
|---|---|
| `TD2_3.ipynb` | K-Means, hierarchical and spectral clustering, evaluation and comparison |
| `TD4_5.ipynb` | PCA, clustering on the reduced data, CCA |
| `data/` | Cleaned dataset (scaled and unscaled) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

## Authors

Amine Benbakreti ([GitHub](https://github.com/Benbakreti0Amine) · [LinkedIn](https://www.linkedin.com/in/mohammed-el-amine-benbakreti-351b51280/)) and Giorgi Svanadze.
