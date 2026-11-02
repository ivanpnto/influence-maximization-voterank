# Influence Maximization with VoteRank-based Methods

Influence maximization consists of choosing *k* seed nodes in a network so that a diffusion process started from them reaches as many nodes as possible. This project implements several VoteRank-based heuristics, combines one of them with Random Walk with Restart, and compares the methods experimentally on a synthetic scale-free network.

## Methods

| Method | Idea |
|---|---|
| **VoteRank** | Each node is scored by the sum of its neighbors' voting ability. After a seed is selected, its neighbors' ability is reduced by 1/⟨k⟩ (⟨k⟩ = average degree), which spreads the seeds apart. |
| **WVoteRank** | Weighted variant: each vote is multiplied by the edge weight. |
| **IWVoteRank** | Also counts 2-hop votes, weighted by the product of the edge weights, and applies a 2-hop ability discount of 1/(2⟨k⟩). |
| **RWR hybrid** | Random Walk with Restart from a seed node (restart probability α = 0.15), multiplied by a WVoteRank-inspired structural score √(degree × weighted degree). |

## Experimental setup

- **Network:** Barabási–Albert graph with 500 nodes and *m* = 3, edge weights drawn from U(0, 1).
- **Diffusion model:** Independent Cascade with a constant propagation probability *p* = 0.05.
- **Spread estimate:** average number of activated nodes over 200 Monte Carlo simulations.
- **Seed budget:** *k* ∈ {5, 10, 15, 20}. Random seed: 42.

## Results

Average influence spread (activated nodes):

| k | VoteRank | WVoteRank | IWVoteRank |
|---|---|---|---|
| 5 | 24.80 | **25.11** | 24.14 |
| 10 | 36.83 | **37.03** | 35.43 |
| 15 | **46.44** | 45.27 | 44.89 |
| 20 | **53.59** | 53.38 | 53.09 |

Seed-selection time (seconds):

| k | VoteRank | WVoteRank | IWVoteRank |
|---|---|---|---|
| 5 | 0.005 | 0.011 | 0.083 |
| 10 | 0.007 | 0.017 | 0.163 |
| 15 | 0.009 | 0.023 | 0.229 |
| 20 | 0.011 | 0.029 | 0.301 |

### Discussion

- **No method clearly dominates on this network.** The spreads differ by less than 2 nodes (about 3%), and the best method changes with *k*. These gaps are within Monte Carlo variability: the two estimates for VoteRank at *k* = 10 in the notebook are 36.50 and 36.83.
- **IWVoteRank's 2-hop exploration is expensive.** It is 17–27× slower than VoteRank and does not improve the spread here.
- **The RWR hybrid** keeps 8 of the 10 top nodes found by pure RWR and reorders the ranking towards structurally central nodes.

**Limitations:** a single synthetic graph and no confidence intervals. Natural next steps are real-world networks, more simulations with confidence intervals, and a greedy (CELF) baseline.

## Project structure

```
├── src/
│   ├── models.py       VoteRank, WVoteRank, IWVoteRank, RWR and the RWR hybrid
│   ├── diffusion.py    Independent Cascade simulation and Monte Carlo spread
│   └── utils.py        Graph helpers
└── notebooks/
    └── evaluation.ipynb   Experiments, tables and plots
```

## Getting started

```bash
git clone https://github.com/ivanpnto/influence-maximization-voterank.git
cd influence-maximization-voterank
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/evaluation.ipynb
```

## License

Released under the [MIT License](LICENSE).
