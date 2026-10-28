checks: {"cos(rebuilt harmful-minus-alpaca, arditi refusal)": 0.705, "cos(refusal, dmu)": -0.363}

A. share of target direction in each 229-dim subspace
| target | persona span | random 229 | Alpaca PCA-229 | span of 229 Alpaca acts |
|---|---|---|---|---|
| refusal | 0.407 | 0.056 ± 0.005 (max 0.075) | 0.344 | 0.319 ± 0.009 (max 0.349) |
| dmu | 0.482 | 0.057 ± 0.005 (max 0.069) | 0.476 | 0.398 ± 0.019 (max 0.437) |
| refusal_rebuilt | 0.691 | 0.056 ± 0.005 (max 0.067) | 0.614 | 0.545 ± 0.020 (max 0.586) |

B. share in persona span of each direction
| direction | share |
|---|---|
| refusal | 0.407 |
| refusal_rebuilt | 0.691 |
| dmu | 0.482 |
| alpaca_mean | 0.584 |
| hb_mean | 0.420 |
| random_gaussian | 0.056 ± 0.005 (max 0.074) |
| shuffled_refusal | 0.436 ± 0.101 (max 0.665) |
| shuffled_dmu | 0.321 ± 0.066 (max 0.523) |
| alpaca_mean_in_random229 | 0.056 ± 0.005 (max 0.071) |
| alpaca_mean_in_alpaca_pca229 | 0.662 |

C. share of centered Alpaca activation variance
| subspace | share |
|---|---|
| persona | 0.393 |
| alpaca_pca229 | 0.915 |
| random229 | 0.056 ± 0.001 (max 0.058) |
