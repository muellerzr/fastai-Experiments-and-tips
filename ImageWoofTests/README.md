# Tests with Mish and Over9000 optimizers on ImageWoof
Both tests used an average of five total runs and recorded is the results from each run, the time per epoch, as well as the learning rate used.


## Five Epochs:
|  | Adam + Mish | Over9000 + ReLU | Over9000 + Mish |
|:--------------:|:-----------:|:---------------:|:---------------:|
| LR | 3e-3 | 1e-2 | 1e-2 |
| Time per Epoch | 2:12 | 2:06 | **1:28** |
| Acc1 | 68.4% | 68.2% | 72.2% |
| Acc2 | 70.8% | 69.4% | 73.8% |
| Acc3 | 67.4% | 68.6% | 74.6% |
| Acc4 | 73.4% | 69.4% | 74.0% |
| Acc5 | 72.2% | 66.6% | 70.8% |
| **Mean** | 70.44% | 68.44% | **73.08%** |
| **Std** | 2.25% | 1.03% | 1.39% |

## Twenty Epochs:

|  | Adam + Mish | Over9000 + ReLU | Over9000 + Mish |
|:--------------:|:-----------:|:---------------:|:---------------:|
| LR | 3e-3 | 1e-2 | 1e-2 |
| Time per Epoch | 2:07 | 2:06 | 2:30 |
| Acc1 | 83.8% | 82.2% | 85.2% |
| Acc2 | 83.2% | 83.4% | 84.2% |
| Acc3 | 84.2% | 82.6% | 85.4% |
| Acc4 | 82.8% | 82.2% | 83.6% |
| Acc5 | 84.8% | 82.4% | 84.8% |
| **Mean** | 83.76% | 82.56% | 84.64% |
| **Std** | 0.70% | 0.44% | 0.66% |
