# GUARD: Graph Universal Adversarial Defense

<p align="center"> <img src="./figs/demo.jpg" /> <p align="center"><em>Fig. 1. An illustrative example of graph universal defense. The universal patch p can be applied to an arbitrary node (here v1) to protect it from adversarial targeted attacks by removing adversarial edges (if exist).</em></p>

# Requirements
+ torch==1.9
+ dgl==0.7.0
+ graphwar

Install [graphwar](https://github.com/EdisonLeeeee/GraphWar):
```bash
git clone https://github.com/EdisonLeeeee/GraphWar.git && cd GraphWar
pip install .
```

# Quick Start
run demo.ipynb

# Reproduce results in our paper
run
```python

python evaluate_guard.py
```
