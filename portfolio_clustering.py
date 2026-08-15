"""
Is My Portfolio Really Diversified?
Grouping S&P 500 stocks by their risk behaviour with K-Means

Turkiye Yapay Zeka Akademisi - Final Medium Project

Purpose:
    Most people diversify by buying stocks from different sectors. I wanted to
    check if that actually works. So instead of using the sector labels, I
    grouped the stocks by how they really behaved in the market (return, risk,
    beta) with K-Means, and then compared my clusters with the sectors.

Problem type:
    Unsupervised learning (clustering). There is no target column, I am
    looking for groups inside the data.

Libraries:
    pandas, numpy - data operations
    matplotlib, seaborn - plots
    scikit-learn - scaling, PCA, K-Means, metrics
    scipy - dendrogram

Plan/program:
    1. Import libraries
    2. Load the price data
    3. Build the risk features for every stock
    4. Add the sector labels
    5. Explore and visualise
    6. Scale the features
    7. Choose k (elbow + silhouette)
    8. K-Means
    9. Look at the clusters
    10. PCA to see the clusters in 2D
    11. Hierarchical clustering as a control
    12. Compare the clusters with the sectors
    13. Test a "diversified" portfolio
    14. Comments

How to run:
    pip install -r requirements.txt
    python portfolio_clustering.py

    The script downloads the data by itself and saves the plots into the
    figures folder.
"""

# 1. Import libraries
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

os.makedirs("figures", exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

PRICES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
SECTORS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


# 2. Load the price data
print("\n--- 2. DATA ---")

if not os.path.exists("prices.csv"):
    print("downloading prices...")
    urllib.request.urlretrieve(PRICES_URL, "prices.csv")
if not os.path.exists("sectors.csv"):
    print("downloading sectors...")
    urllib.request.urlretrieve(SECTORS_URL, "sectors.csv")

prices = pd.read_csv("prices.csv", parse_dates=["date"])

print("S&P 500 daily prices.")
print(f"Rows: {prices.shape[0]}, columns: {prices.shape[1]}")
print(f"Date range: {prices.date.min().date()} - {prices.date.max().date()}")
print(f"Number of stocks: {prices.Name.nunique()}")
print(prices.head())

print("\nMissing values:")
print(prices.isna().sum())

# I put the closing prices into a wide table, one column per stock.
wide = prices.pivot(index="date", columns="Name", values="close")

# Some stocks do not have the full history (they joined the index later).
# I keep only the ones with at least 98% of the days, and fill the small gaps.
before = wide.shape[1]
wide = wide.dropna(axis=1, thresh=int(len(wide) * 0.98)).ffill()
print(f"\nStocks with an almost complete history: {wide.shape[1]} (dropped {before - wide.shape[1]})")

returns = wide.pct_change().dropna()
market = returns.mean(axis=1)  # equal weighted market, I use it for beta
print(f"Daily returns table: {returns.shape[0]} days x {returns.shape[1]} stocks")


# 3. Build the risk features
print("\n--- 3. FEATURE ENGINEERING ---")

# The raw data is just prices, so I cannot cluster it directly. I calculate
# 6 numbers for every stock that describe how it behaved in these 5 years.
rows = []
for ticker in returns.columns:
    r = returns[ticker]
    px = wide[ticker]
    years = (wide.index[-1] - wide.index[0]).days / 365.25

    cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1   # yearly return
    volatility = r.std() * np.sqrt(252)                    # yearly risk
    max_drawdown = ((px - px.cummax()) / px.cummax()).min() # worst fall
    beta = np.cov(r, market)[0, 1] / np.var(market)        # market sensitivity
    market_corr = r.corr(market)                           # co-movement
    downside_vol = r[r < 0].std() * np.sqrt(252)           # only bad days

    rows.append({
        "Symbol": ticker,
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "market_corr": market_corr,
        "downside_vol": downside_vol,
    })

df = pd.DataFrame(rows)
print(f"Feature table: {df.shape[0]} stocks, {df.shape[1] - 1} features")
print(df.head())


# 4. Add the sector labels
print("\n--- 4. SECTORS ---")

sectors = pd.read_csv("sectors.csv")[["Symbol", "GICS Sector"]]
df = df.merge(sectors, on="Symbol", how="left")

# Some tickers are not in the sector list anymore, because those companies
# left the index after 2018. I drop them, since I need the sector to make my
# comparison. This gives a survivorship bias and I write about it at the end.
missing_sector = df["GICS Sector"].isna().sum()
df = df.dropna(subset=["GICS Sector"]).reset_index(drop=True)
print(f"Dropped {missing_sector} stocks without a sector label.")
print(f"Remaining: {len(df)} stocks")
print(df["GICS Sector"].value_counts())


# 5. Explore and visualise
print("\n--- 5. EXPLORATION ---")
print(df.describe().T.round(3))

features = ["cagr", "volatility", "max_drawdown", "beta", "market_corr", "downside_vol"]

# Risk and return together. This is the classic picture every investor knows.
plt.figure(figsize=(9, 6))
plt.scatter(df["volatility"], df["cagr"], alpha=0.6, s=30, color="steelblue")
plt.axhline(0, color="grey", lw=1)
plt.xlabel("Volatility (yearly risk)")
plt.ylabel("CAGR (yearly return)")
plt.title("Risk and return of S&P 500 stocks, 2013-2018")
plt.tight_layout()
plt.savefig("figures/01_risk_return.png", dpi=120)
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between the features")
plt.tight_layout()
plt.savefig("figures/02_feature_correlation.png", dpi=120)
plt.close()

df[features].hist(bins=40, figsize=(12, 7), color="steelblue", edgecolor="black")
plt.suptitle("Distribution of the features")
plt.tight_layout()
plt.savefig("figures/03_distributions.png", dpi=120)
plt.close()


# 6. Scaling
print("\n--- 6. SCALING ---")

# K-Means uses distances, so the features must be on the same scale. Without
# scaling, beta (around 1) would be invisible next to max_drawdown (around
# -0.36) and volatility (around 0.24) would dominate everything.
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
print("Mean after scaling:", np.round(X.mean(axis=0), 3))
print("Std after scaling:", np.round(X.std(axis=0), 3))


# 7. Choosing k
print("\n--- 7. HOW MANY CLUSTERS? ---")

inertias = []
silhouettes = []
k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"k={k}  inertia={km.inertia_:8.1f}  silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_values), inertias, "o-", color="steelblue")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow method")
axes[1].plot(list(k_values), silhouettes, "o-", color="darkorange")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score")
plt.tight_layout()
plt.savefig("figures/04_choosing_k.png", dpi=120)
plt.close()

# The silhouette score is the highest at k=2, but k=2 only separates "risky"
# and "safe" stocks and that is not useful for me. The elbow bends around 4
# and at k=4 the groups make sense when I look at them. So I choose k=4 and
# accept a slightly lower score for a more meaningful result.
K = 4


# 8. K-Means
print(f"\n--- 8. K-MEANS WITH k={K} ---")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())
print(f"Silhouette score: {silhouette_score(X, df['cluster']):.4f}")


# 9. Look at the clusters
print("\n--- 9. CLUSTER PROFILES ---")

profiles = df.groupby("cluster")[features].mean().round(3)
print(profiles.to_string())

# I gave the clusters names by looking at these averages.
names = {
    0: "Steady compounders",
    1: "High beta cyclicals",
    2: "High risk, low reward",
    3: "Defensive / low beta",
}
df["cluster_name"] = df["cluster"].map(names)
print("\nMy names for them:")
for i, n in names.items():
    print(f"  Cluster {i}: {n}")

plt.figure(figsize=(10, 5))
sns.heatmap(
    df.groupby("cluster")[features].mean().T,
    annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
)
plt.xlabel("Cluster")
plt.title("Average feature values per cluster")
plt.tight_layout()
plt.savefig("figures/05_cluster_profiles.png", dpi=120)
plt.close()

plt.figure(figsize=(9, 6))
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    plt.scatter(sub["volatility"], sub["cagr"], label=f"{c}: {names[c]}", alpha=0.7, s=35)
plt.xlabel("Volatility")
plt.ylabel("CAGR")
plt.title("The clusters on the risk-return map")
plt.legend()
plt.tight_layout()
plt.savefig("figures/06_clusters_risk_return.png", dpi=120)
plt.close()


# 10. PCA
print("\n--- 10. PCA ---")

# I have 6 features so I cannot draw them. PCA squeezes them into 2 dimensions
# so I can look at the clusters.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_.round(3)}")
print(f"Total: {pca.explained_variance_ratio_.sum():.1%}")

plt.figure(figsize=(9, 7))
for c in sorted(df["cluster"].unique()):
    mask = df["cluster"] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"{c}: {names[c]}", alpha=0.7, s=35)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
plt.title("Clusters after PCA")
plt.legend()
plt.tight_layout()
plt.savefig("figures/07_pca_clusters.png", dpi=120)
plt.close()


# 11. Hierarchical clustering as a control
print("\n--- 11. HIERARCHICAL CLUSTERING ---")

agg = AgglomerativeClustering(n_clusters=K, linkage="ward")
df["agg_cluster"] = agg.fit_predict(X)

agreement = adjusted_rand_score(df["cluster"], df["agg_cluster"])
print(f"Agreement with K-Means (Adjusted Rand Index): {agreement:.4f}")
# If a completely different algorithm finds similar groups, then the groups
# are probably really in the data and not something K-Means invented.

plt.figure(figsize=(12, 5))
dendrogram(linkage(X, method="ward"), no_labels=True, color_threshold=None)
plt.title("Dendrogram")
plt.xlabel("Stocks")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("figures/08_dendrogram.png", dpi=120)
plt.close()


# 12. Clusters vs sectors
print("\n--- 12. CLUSTERS VS SECTORS ---")

contingency = pd.crosstab(df["GICS Sector"], df["cluster"], normalize="index") * 100
print("How each sector spreads over the clusters (%):")
print(contingency.round(0).to_string())

ari = adjusted_rand_score(df["GICS Sector"], df["cluster"])
print(f"\nAdjusted Rand Index between sector and cluster: {ari:.4f}")
print("(1 = they say the same thing, 0 = they are unrelated)")

spread = (pd.crosstab(df["GICS Sector"], df["cluster"]) > 0).sum(axis=1)
print("\nHow many clusters each sector spreads into:")
print(spread.sort_values(ascending=False).to_string())

plt.figure(figsize=(10, 6))
sns.heatmap(contingency, annot=True, fmt=".0f", cmap="Blues")
plt.xlabel("Cluster")
plt.ylabel("")
plt.title("Sector vs cluster (% of the sector)")
plt.tight_layout()
plt.savefig("figures/09_sector_vs_cluster.png", dpi=120)
plt.close()


# 13. Testing a "diversified" portfolio
print("\n--- 13. IS THIS PORTFOLIO DIVERSIFIED? ---")

# A portfolio built the classic way: one stock from 8 different sectors.
portfolio = ["AAPL", "JPM", "JNJ", "PG", "XOM", "DIS", "BA", "NEE"]
selected = df[df["Symbol"].isin(portfolio)]

print("A portfolio with one stock from 8 different sectors:")
print(selected[["Symbol", "GICS Sector", "cluster_name", "cagr", "volatility", "beta"]]
      .sort_values("cluster_name").round(3).to_string(index=False))

print(f"\nDifferent sectors: {selected['GICS Sector'].nunique()} out of 8")
print(f"Different clusters: {selected['cluster'].nunique()} out of {K}")


# 14. Comments
print("\n--- 14. MY COMMENTS ---")
print(f"""
What I found:
    K-Means found 4 groups that make sense to me:
      - Steady compounders: good return, low risk, moves with the market
      - High beta cyclicals: more return but bigger falls
      - High risk low reward: very volatile and it did not pay off (mostly energy)
      - Defensive: low beta, small falls, mostly utilities, staples, real estate

    The Adjusted Rand Index between the sectors and my clusters is only {ari:.2f},
    so the sector of a stock says almost nothing about how it behaves.

    Almost every sector spreads into 3 or 4 different clusters. Only the
    defensive sectors are consistent, around 85-90% of utilities, real estate
    and consumer staples are in the same cluster.

    My 8 stock portfolio has 8 different sectors but only 2 different
    clusters. So it looks diversified but it is not really.

Limitations:
    - The data is 2013-2018 and it was mostly a rising market. In a crisis
      the groups would probably look different.
    - I removed the companies that left the index after 2018, so I only look
      at the survivors. The real "high risk" group is bigger than mine.
    - I used the sector labels of today for old prices, some companies
      changed sector in the meantime.
    - K-Means always finds clusters even when there are none, so I checked
      with hierarchical clustering and the result was similar.
    - The silhouette score was low, which means the groups are not perfectly
      separated. Stocks are a continuum, not real separate boxes.
""")

print("Finished. The plots are in the figures folder.")
