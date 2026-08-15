# Is My Portfolio Really Diversified? I Asked K-Means Instead of My Broker

*Grouping 339 S&P 500 stocks by how they actually behaved — and finding out that sector labels lie.*

---

Every beginner investing guide gives you the same advice: don't put all your money in one sector. Buy some tech, some banking, some healthcare, some energy. Spread it around. That's diversification.

I followed that advice. And then I started learning machine learning, and a question started bothering me:

**What if two stocks from completely different sectors behave exactly the same way?**

If Apple and a utility company both go down 20% in the same week, it doesn't really matter that one is labelled "Information Technology" and the other "Utilities". They're the same position as far as my money is concerned.

Sector is a label somebody assigned based on what a company *sells*. It says nothing about how the stock *moves*. So for my final project I decided to ignore the labels completely, group stocks by their actual market behaviour using K-Means, and then compare my groups against the official sectors.

The short version: they barely agree at all. And the "diversified" portfolio I built to test it turned out to contain 8 sectors but only 2 real behaviour groups.

Here's how I got there.

---

## The data

I used two public datasets:

- **Daily prices for S&P 500 stocks**, February 2013 to February 2018 — 619,040 rows of open/high/low/close/volume for 505 companies.
- **The S&P 500 constituents list**, which gives me the official GICS sector for each ticker.

```python
prices = pd.read_csv("prices.csv", parse_dates=["date"])
print(prices.shape)          # (619040, 7)
print(prices.Name.nunique()) # 505
```

Five years, about 1,258 trading days per stock. It's not a huge dataset by industry standards but it's plenty for what I wanted.

### A trap I nearly walked into

My original plan was fancier. I found a file with S&P 500 fundamentals — P/E ratio, price/book, dividend yield, market cap — and I was going to cluster on valuation *and* risk together.

Then I actually looked at the numbers:

```
Symbol  Price   Price/Earnings   Market Cap
AAPL    305.93  35.04            4.46e+12
NVDA    225.16  34.48            5.45e+12
```

Apple at a $4.4 trillion market cap. That's a **current** file. My prices end in February 2018, when Apple was worth about $900 billion.

If I had merged them, I would have been clustering 2013–2018 price behaviour against 2026 valuations — using information from the future to explain the past. Classic look-ahead bias, and my results would have looked fine while being completely meaningless.

I threw the fundamentals away and used only the price data. This cost me some interesting features, but everything left comes from one source and one time period.

I think this is worth saying out loud, because nothing in the file itself warns you. Two datasets can merge perfectly on a key column and still be nonsense together.

---

## Turning prices into behaviour

Here's the actual problem: you can't cluster a price series. K-Means needs one row per stock with a fixed set of numbers. I had 1,258 daily prices per stock instead.

So the real work of this project was feature engineering — compressing five years of price movement into six numbers that describe *how a stock behaves*:

```python
cagr         = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1
volatility   = r.std() * np.sqrt(252)
max_drawdown = ((px - px.cummax()) / px.cummax()).min()
beta         = np.cov(r, market)[0, 1] / np.var(market)
market_corr  = r.corr(market)
downside_vol = r[r < 0].std() * np.sqrt(252)
```

What each one means:

| Feature | What it tells me |
|---|---|
| **CAGR** | Yearly return. How much did it actually make? |
| **Volatility** | Yearly standard deviation. How bumpy was the ride? |
| **Max drawdown** | The worst peak-to-bottom fall. How much pain at the worst moment? |
| **Beta** | Sensitivity to the market. Beta 1.2 means it moves 20% more than the market. |
| **Market correlation** | How closely it follows the market at all. |
| **Downside volatility** | Volatility calculated only on losing days. |

That last one matters more than it looks. Normal volatility punishes a stock for going *up* fast, which as an investor I don't mind at all. Downside volatility only measures the movement I actually care about.

For beta I needed a market benchmark, so I built an equal-weighted index from all the stocks:

```python
market = returns.mean(axis=1)
```

Not perfect — the real S&P 500 is cap-weighted — but consistent across every stock, which is what matters for comparison.

I dropped stocks without an almost-complete price history (476 survived), then merged in the sectors. 137 tickers weren't in the current constituents list because those companies left the index after 2018, and I need the sector for my comparison, so I dropped them too.

**Final dataset: 339 stocks, 6 features.** That drop introduces survivorship bias and I come back to it at the end.

---

## Scaling (the step you can't skip)

```python
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
```

K-Means works by measuring distances between points. My features live on completely different scales — beta sits around 1.0, max drawdown around -0.36, volatility around 0.24.

Without scaling, the features with bigger numbers dominate the distance calculation and the small ones become invisible. The algorithm would essentially cluster on beta alone and ignore drawdown. Scaling puts everything at mean 0, standard deviation 1, so each feature gets an equal vote.

---

## How many clusters?

This is the awkward part of K-Means: you have to tell it how many groups to find, before you know how many there are.

I used the two standard methods.

**![Figure: elbow method and silhouette score]**

```
k=2   inertia=1378.8   silhouette=0.3328
k=3   inertia=1051.8   silhouette=0.2769
k=4   inertia= 890.7   silhouette=0.2536
k=5   inertia= 763.0   silhouette=0.2741
k=6   inertia= 684.1   silhouette=0.2759
```

The silhouette score says **k=2** is best. And technically it's right — the cleanest split in the data is just "risky stocks" and "safe stocks".

But that's not useful to me. I already knew stocks come in risky and safe. Splitting 339 companies into two giant piles doesn't help me build a portfolio.

The elbow bends around 4, and when I looked at what k=4 actually produced, the groups made sense as things I could describe in words. So **I chose k=4 and accepted a lower silhouette score in exchange for a more useful answer.**

I want to be honest that this is a judgement call and not a mathematical result. The metric pointed one way and I went another way on purpose. In unsupervised learning there's no accuracy score to hide behind — you have to defend your choice, and "it gave me groups I could interpret" is a legitimate defence as long as you say it out loud.

---

## The four groups

```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)
```

| Cluster | CAGR | Volatility | Max DD | Beta | Mkt corr | n |
|---|---:|---:|---:|---:|---:|---:|
| **0 — Steady compounders** | 17.5% | 0.192 | -22% | 0.96 | 0.63 | 102 |
| **1 — High beta cyclicals** | 14.7% | 0.260 | -39% | 1.20 | 0.59 | 95 |
| **2 — High risk, low reward** | 8.2% | 0.383 | -62% | 1.26 | 0.42 | 35 |
| **3 — Defensive / low beta** | 10.2% | 0.197 | -27% | 0.67 | 0.43 | 107 |

The names are mine, assigned after looking at the averages.

**Cluster 0 — Steady compounders.** The best group by a distance. Highest return, lowest volatility, smallest drawdown. They track the market closely (correlation 0.63) and just quietly compound.

**Cluster 1 — High beta cyclicals.** Beta 1.20, so they amplify the market. Decent returns but a -39% average worst fall. You get paid for the risk, but not generously.

**Cluster 2 — High risk, low reward.** The group nobody wants. Highest volatility (0.383), worst drawdowns (-62%), and the *lowest* returns of all four at 8.2%. Half of this cluster is Energy — this window covers the 2014–2016 oil crash. A useful reminder that risk and return are supposed to go together in theory, and regularly don't in practice.

**Cluster 3 — Defensive / low beta.** Beta 0.67, meaning they only move two-thirds as much as the market. Modest returns, small drawdowns. The sleep-well-at-night group.

**![Figure: clusters on the risk-return map]**

### Seeing it in 2D

Six features is four too many to plot, so I used PCA to squeeze them into two dimensions:

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)  # [0.521 0.258]
```

Two components hold **77.8%** of the original information, which is enough to trust the picture.

**![Figure: PCA scatter plot of the clusters]**

The clusters separate cleanly. PC1 (52%) runs left to right and is basically an overall risk axis — cluster 2 sits far out on the right, alone. PC2 (26%) separates the defensive stocks at the bottom from the market-tracking ones at the top.

### Checking that the clusters are real

K-Means will *always* return clusters. Ask for four groups in pure random noise and you get four groups. So I ran a completely different algorithm on the same data:

```python
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
adjusted_rand_score(df["cluster"], df["agg_cluster"])  # 0.5328
```

Hierarchical clustering agrees with K-Means at an Adjusted Rand Index of **0.53**. Not identical, but far above the ~0 you'd get from two random labellings. Two different algorithms finding similar structure is decent evidence the structure is actually in the data.

---

## The main result: sectors don't describe behaviour

Now the comparison I built the whole project for.

```python
adjusted_rand_score(df["GICS Sector"], df["cluster"])  # 0.0915
```

**0.09.** On a scale where 1 means "these two labellings say the same thing" and 0 means "unrelated".

Knowing a stock's sector tells you almost nothing about how it behaves.

**![Figure: sector vs cluster heatmap]**

Reading the heatmap row by row is where it gets interesting:

- **Information Technology** splits 39% / 39% / 13% / 8% across all four clusters. "I own tech" describes your risk exposure not at all.
- **Health Care** spreads 30 / 25 / 9 / 36. Also everywhere.
- **Financials** are 55% steady compounders and 39% high-beta cyclicals — two very different profiles inside one label.
- **Energy** is the only sector that concentrates in the bad cluster: 50% in "high risk, low reward", 44% in high-beta.

Every single sector spreads across at least three of my four clusters.

**But here's the nuance that made me trust the result more:** the defensive sectors are genuinely coherent. Consumer Staples is **90%** cluster 3. Real Estate is **88%**. Utilities is **85%**.

So sector labels aren't useless — they work well for the defensive end of the market, where the label really does describe the behaviour. They fall apart everywhere else. If the answer had been "labels are always wrong" I'd have suspected my own analysis. "Labels work in one specific place and fail elsewhere" is a more believable result.

---

## Testing an actual portfolio

Time to put my own advice on trial. I built the most textbook-diversified portfolio I could — one stock from eight different sectors:

```python
portfolio = ["AAPL", "JPM", "JNJ", "PG", "XOM", "DIS", "BA", "NEE"]
```

| Symbol | Sector | Cluster | CAGR | Volatility | Beta |
|---|---|---|---:|---:|---:|
| BA | Industrials | Steady compounders | 35.4% | 0.214 | 1.02 |
| DIS | Communication Services | Steady compounders | 13.9% | 0.184 | 0.91 |
| JNJ | Health Care | Steady compounders | 11.8% | 0.143 | 0.66 |
| JPM | Financials | Steady compounders | 18.4% | 0.204 | 1.19 |
| XOM | Energy | Steady compounders | -2.8% | 0.175 | 0.87 |
| AAPL | Information Technology | Defensive / low beta | 18.7% | 0.232 | 0.81 |
| NEE | Utilities | Defensive / low beta | 15.5% | 0.164 | 0.51 |
| PG | Consumer Staples | Defensive / low beta | 1.6% | 0.142 | 0.55 |

**8 sectors. 2 clusters.**

By the standard advice this portfolio is perfectly diversified. By actual behaviour it's two bets wearing eight different hats.

My favourite detail: **Apple ended up in the defensive cluster.** With a beta of 0.81 over this period, Apple moved *less* than the market — it behaved more like Procter & Gamble than like a growth stock. If you'd bought AAPL as your "aggressive tech position" and NEE as your "safe utility", the data says you bought two versions of the same thing.

---

## What I'd tell myself before starting

**Feature engineering was the whole project.** The clustering itself is four lines of code. Turning 1,258 daily prices into six numbers that mean something took the vast majority of the work, and every interesting thing in the results came from choosing those six numbers well.

**Unsupervised learning has no safety net.** With classification you get an accuracy score and you know where you stand. Here I chose k myself, named the clusters myself, and decided myself whether the result was meaningful. That freedom is uncomfortable — and it's why the validation steps (silhouette, PCA, the hierarchical cross-check) matter so much more than they would in a supervised project.

**The data trap was the most valuable lesson.** Catching that market cap figure was luck as much as skill. Now I check the vintage of every file before merging anything.

---

## Limitations

I want to be clear about what this analysis can't support:

1. **The period was a bull market.** February 2013 to February 2018 was mostly rising. In a real crisis correlations tend to go to 1 and the clusters would probably collapse into each other. My "defensive" group is only defensive in the conditions I tested.

2. **Survivorship bias.** I dropped 137 stocks that left the index after 2018 — and companies leave the index mostly by doing badly. My "high risk, low reward" cluster of 35 should almost certainly be bigger.

3. **Old sector labels on old prices.** I used today's GICS sectors for 2013–2018 behaviour. Some companies changed sector in between — Google and Facebook moved to Communication Services in 2018.

4. **The silhouette score is low (0.25).** The clusters aren't cleanly separated, and looking at the PCA plot the boundaries genuinely are fuzzy. Stocks are a continuum, not four tidy boxes. The groups are useful, not real.

5. **This is not investment advice.** It's an analysis of five years of history with six hand-picked features. Past behaviour is exactly that.

---

## Conclusion

I started with a question I couldn't answer by reading my portfolio: is diversifying across sectors real diversification?

The answer from the data is **mostly no**. Adjusted Rand Index of 0.09 between sector labels and behaviour clusters. Every sector spread across three or four different risk profiles. A textbook eight-sector portfolio that turned out to be two positions in disguise.

The version of this I'd actually use: stop asking *"do I own different industries?"* and start asking *"do I own different behaviours?"* Those are not the same question, and only one of them shows up on a brokerage screen.

The whole thing is K-Means, PCA and StandardScaler — the same tools from week two of the bootcamp. What made it work wasn't the algorithm, it was spending the time to turn raw prices into features that meant something.

---

*Code and charts: [GitHub repository link]*

*Data: S&P 500 daily prices 2013–2018, and the S&P 500 constituents list, both publicly available.*
