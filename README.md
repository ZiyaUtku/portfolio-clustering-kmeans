# Is My Portfolio Really Diversified?

Clustering S&P 500 stocks by their risk behaviour with K-Means, and comparing
the result against the official GICS sectors.

Final Medium project - Turkiye Yapay Zeka Akademisi.

## What this does

Sector labels describe what a company sells, not how its stock moves. This
project ignores the labels, builds 6 risk features for every stock from 5
years of daily prices, clusters them with K-Means, and checks whether the
clusters agree with the sectors.

They mostly do not. The Adjusted Rand Index between sector and cluster is
**0.09**, and a portfolio with one stock from 8 different sectors turned out
to sit in only 2 of the 4 behaviour clusters.

## Data

- S&P 500 daily prices, Feb 2013 - Feb 2018 (619,040 rows, 505 stocks)
- S&P 500 constituents list, for the GICS sector labels

Both are downloaded automatically by the script, so they are not in the repo.

## Method

1. Build 6 features per stock: CAGR, volatility, max drawdown, beta,
   market correlation, downside volatility
2. StandardScaler
3. Choose k with the elbow method and the silhouette score
4. K-Means with k=4
5. PCA for the 2D visualisation
6. Hierarchical clustering as a control (ARI 0.53 with K-Means)
7. Compare the clusters against the GICS sectors

## Results

| Cluster | CAGR | Volatility | Max DD | Beta | n |
|---|---:|---:|---:|---:|---:|
| Steady compounders | 17.5% | 0.192 | -22% | 0.96 | 102 |
| High beta cyclicals | 14.7% | 0.260 | -39% | 1.20 | 95 |
| High risk, low reward | 8.2% | 0.383 | -62% | 1.26 | 35 |
| Defensive / low beta | 10.2% | 0.197 | -27% | 0.67 | 107 |

## How to run

```bash
pip install -r requirements.txt
python portfolio_clustering.py
```

The plots are saved into the `figures` folder.

## Files

- `portfolio_clustering.py` - the whole analysis
- `requirements.txt`
- `figures/` - the 9 charts
