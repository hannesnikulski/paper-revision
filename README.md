# Paper Revision: Discovering parametrizations of implied volatility with symbolic regression

## Data Analysis and preprocessing

The IV-Data has the following structure:

| Column Name | Description | 
|--------------|--------------|
| Date | Current Date. Ranges from 1996-01-04 to 2025-09-08 (7465 days) |
| Expiry | Date of expiry | 
| Texp | Time to expiry in years (difference between 'Date' and 'Expiry') | 
| Strike | Strike price | 
| Bid | Bid-price  | 
| Ask | Ask-price   | 
| Fwd | Forward-price | 
| CallMid | Implied Volatility in basis points (suspected) | 

Calculated Columns:

| Column Name | Description | 
|--------------|--------------|
| z | Log-moneyness, calculated as $z = \ln\left(\frac{\text{Strike}}{\text{Fwd}}\right)$ |
| w | Total implied volatility, calculated as $w = \text{CallMid}^2 \cdot \text{Texp}$ | 

**For the following data analysis the considered time period is between 2000-01-01 and 2024-12-31.**

Per (Date, Texp) we obtain an IV-slice. The number of observations per slice varies largely, as can be seen in the table below. In total we obtain 143,680 slices and 18,752,333 total observations. 

```plaintext
count    143680
mean        130
std          89
min           6
25%          62
50%         115
75%         170
max         499
dtype: int64
```

Using all slices in a naive Custering would lead to the following sizes for the Distance matrix using various data types.

| Data Type | Full Distance Matrix | Upper Right Distance Matrix |
|-----------|----------------------|-----------------------------|
| Float 64  | 165 GB               | 82 GB                       |
| Float 32  | 82 GB                | 41 GB                       |
| Float 16  | 41 GB                | 20 GB                       |

### Challenges

Calculating the full distance matrix might be infeasible depending on the memory constraints of the HPC. (Note: While it is only necessary to calculate only one half of the distance matrix it is neccessary to generate the full matrix for the clustering algorithms).

An second challenge is the non-alignment of the z-values between slices.

### Proposals / Options for challenge 1

##### Option 1 (as last time)
Compute the full distance matrix for clustering with a custom distance function where two slices are only considered where their domains intersect. If we hit memory limits on the HPC we reduce the number of slices by considering a more selective subset of the data.

##### Option 1.2
We group the data by common maturities (e.g. <1 week, 1 month, 3 months, 6 months, 1 year, >1 year) and compute clusters on each group.

```plaintext
Maturity        Number of slices
<=1wk           10226
  1m            31418
  3m            31052
  6m            21356
  1y            27034
 >1y            22594
```

The distance matrix would then only be about 8 GB of size.

##### Option 2

The [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) alogrithm allows for fast and memory efficient clustering. The algorithm works by repeatedly taking random subsamples of the dataset and updating the clusters until convergence. Apparently the sacrifice in the cluster precision is only minor. (Time on my PC on the full dataset: ~1s)

The algorithm needs aligned z-values. Therefore I propose the following approaches.
 - Normalize the z-values to a common range (e.g. [0, 1]) and interpolate on a common grid. Then for all candidate functions $x \mapsto f(x)$ consider the functions $x \mapsto f(ax + b)$ with two additional parameters.
 - create a common grid for all slices and interpolate on this. If the interpolation points are outside of the range of the slice the interpolation is filled with a distinct value (e.g. 0 or -1). After that scale the w-values by the number of valid data-points per interpolated z-value.

The second option will probably result in rather odd looking clusters as the ranges of the individual slices varies quite dramatically. Therefore the range where many slices intersect will be small.

## Arbitrage Analysis

| Maturity Group | Number of Slices with Arb. | Pct. negative Area (indicator from the paper) |
|----------------|----------------------------|-----------------------------------------------|
| <=1wk          | 104 / 10299                | 0.0                                           |
| 1m             | 51 / 31400                 | 1e-8                                          |
| 3m             | 253 / 30989                | 1e-5                                          |
| 6m             | 2928 / 21370               | 0.0002                                        |
| 1y             | 7399 / 27025               | 0.0016                                        |
| >1y            | 8373 / 22597               | 0.0190                                        |

## Removed curves with inside NaN values

| Maturity Group | Number of Slices with Arb. |
|----------------|----------------------------|
| <=1wk          | 10299 ->  4170             |
| 1m             | 31400 -> 22782             |
| 3m             | 30989 -> 22423             |
| 6m             | 21370 -> 16914             |
| 1y             | 27025 -> 22908             |
