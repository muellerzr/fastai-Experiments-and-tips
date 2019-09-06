# Experiments and Results

## Adults
10 runs of 5 epochs

| Shorthand Name | Schedule Type | Opimizer | Mish (1/0) | Mean | Standard Deviation | Maximum | Minimum | P-Value |
|:--------------:|:------------------:|:----------:|:----------:|:------:|:------------------:|:-------:|:-------:|:-------:|
| Model 1 | One Cycle | Adam | 0 | 0.8385 | 0.006258 | 0.845 | 0.83 | NULL |
| Model 2 | One Cycle | Adam | 1 | 0.836 | 0.00459 | 0.845 | 0.83 | N/A |
| Model 3 | Flatten and Anneal | RangerLars | 0 | 0.8385 | 0.00474 | 0.845 | 0.83 | 1 |
| Model 4 | Flatten and Anneal | RangerLars | 1 | 0.835 | 0.00623 | 0.845 | 0.825 | N/A |
| Model 5 | Flatten and Anneal | Ranger | 0 | 0.848 | 0.01059 | 0.86 | 0.83 | 0.0144 |
| Model 6 | Flatten and Anneal | Ranger | 1 | 0.834 | 0.00658 | 0.845 | 0.825 | N/A |

