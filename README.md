## multi-armed-bandit

This repo is set up for a blog post I wrote on ["The Multi-Armed Bandit Problem and Its Solutions"](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html).

---

The result of a small experiment on solving a Bernoulli bandit with K = 10 slot machines, each with a randomly initialized reward probability.

![Alt text](results_K10_N5000.png?raw=true "K=10 N=5000")

- (Left) The plot of time step vs the cumulative regrets.
- (Middle) The plot of true reward probability vs estimated probability.
- (Right) The fraction of each action is picked during the 5000-step run.
