# Automatic conversation behaviro annotator

## Setup

To install the required packages, run:

    pip install -r requirements.txt



## Table on random seeds 

| scale | seed | name     |
|-------|------|----------|
| 1e5   | 1    | dkwnlvzm |
| 1e5   | 2    | 95f8k8zc |
| 1e5   | 3    | 967ufsfk |
| 1e6   | 1    | lb86b69m |
| 1e6   | 2    | 5z07yaqp |
| 1e6   | 3    | he3nnzld |
| 1e7   | 1    | qpp61q7x |
| 1e7   | 2    | m6s9vokb |
| 1e7   | 3    | uu5rtja8 |


## Experiment design
Dataset scale: 1e5, 1e6, 1e7

Pretrain seeds: 1, 2, 3

Finetune seeds: 3, 999, 1024

Generation seeds: 1, 2, 3