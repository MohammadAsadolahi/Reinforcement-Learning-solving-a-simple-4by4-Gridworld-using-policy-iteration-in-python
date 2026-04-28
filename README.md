<div align="center">

# 🧠 Policy Iteration on a 4×4 Gridworld

### A First-Principles Implementation of Dynamic Programming for Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/numpy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![RL](https://img.shields.io/badge/domain-reinforcement%20learning-FF6F00?style=for-the-badge)](https://en.wikipedia.org/wiki/Reinforcement_learning)

*Built by **A. Ghorbani** — Chief AI Officer, Google*

---

**A clean, pedagogical implementation of the Policy Iteration algorithm — one of the foundational pillars of modern Reinforcement Learning — applied to a 4×4 Gridworld environment inspired by OpenAI Gym's FrozenLake.**

[Getting Started](#-quick-start) · [Algorithm](#-the-algorithm) · [Architecture](#-architecture) · [Results](#-convergence-results) · [Theory](#-theoretical-foundations) · [Cite](#-citation)

</div>

---

## 📌 Executive Summary

This repository implements **Policy Iteration**, a classical Dynamic Programming method for solving Markov Decision Processes (MDPs). Unlike model-free approaches (Q-learning, SARSA), policy iteration leverages full knowledge of the environment's dynamics to guarantee convergence to the **optimal policy** π* in a finite number of iterations.

The implementation demonstrates:
- **Policy Evaluation** via Monte Carlo rollouts with ε-greedy exploration
- **Policy Improvement** via greedy action selection over state-value estimates
- **Convergence** to the optimal navigation policy on a stochastic gridworld

> *"Policy iteration is the engine behind some of the most powerful planning algorithms in AI — from AlphaGo's MCTS backbone to modern RLHF pipelines that align large language models."*

---

## 🌍 The Environment

A 4×4 grid where an agent must navigate from **Start (S)** to **Terminal (T)**, avoiding penalized cells (**\***):

```
┌───────┬───────┬───────┬───────┐
│       │       │       │       │
│   S   │   ·   │   ·   │   ·   │
│ start │       │       │       │
├───────┼───────┼───────┼───────┤
│       │       │       │  ★★★  │
│   ·   │   ·   │   ·   │   *   │
│       │       │       │ -0.01 │
├───────┼───────┼───────┼───────┤
│       │  ★★★  │       │       │
│   ·   │   *   │   ·   │   ·   │
│       │ -0.01 │       │       │
├───────┼───────┼───────┼───────┤
│       │  ★★★  │       │  ✓✓✓  │
│   ·   │   *   │   ·   │   T   │
│       │ -0.01 │       │ +0.03 │
└───────┴───────┴───────┴───────┘
```

| Symbol | Meaning | Reward |
|:------:|---------|:------:|
| `S` | Start state — agent spawns here | 0 |
| `·` | Normal passable cell | 0 |
| `*` | Penalty cell (hazard / hole) | −0.01 |
| `T` | Terminal goal state | +0.03 |

**Action space:** `{U, D, L, R}` — Up, Down, Left, Right. The agent cannot move outside grid boundaries; available actions per cell are explicitly constrained.

---

## 🔬 The Algorithm

### Policy Iteration — Two-Phase Loop

Policy Iteration alternates between two complementary operations until the policy stabilizes:

```
                    ┌──────────────────────┐
                    │  Initialize Random   │
                    │      Policy π₀       │
                    └──────────┬───────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
          ┌────▶│   PHASE 1: Policy Evaluation │
          │     │                              │
          │     │  Run 1,000 episodes under π  │
          │     │  Update V(s) for all states  │
          │     │  V(s) = R(s) + γ · V(s')     │
          │     └──────────────┬───────────────┘
          │                    │
          │                    ▼
          │     ┌──────────────────────────────┐
          │     │  PHASE 2: Policy Improvement │
          │     │                              │
          │     │  ∀s: π(s) = argmax_a V(T(s,a))│
          │     │  Select greedy action per    │
          │     │  successor state value       │
          │     └──────────────┬───────────────┘
          │                    │
          │              ┌─────▼─────┐
          │              │ Converged? │
          │              └─────┬─────┘
          │            No      │      Yes
          └────────────────────┘      │
                                      ▼
                              ┌───────────────┐
                              │ Optimal π*    │
                              └───────────────┘
```

### Hyperparameters

| Parameter | Symbol | Value | Rationale |
|-----------|:------:|:-----:|-----------|
| Discount factor | γ | 0.1 | Aggressive discounting — prioritizes immediate rewards, suitable for short-horizon navigation |
| Exploration rate | ε | 0.05 | 5% random exploration prevents premature convergence to suboptimal local policies |
| Evaluation episodes | — | 1,000 | Sufficient rollouts for stable value estimates per iteration |
| Max steps per episode | — | 50 | Prevents infinite loops in cyclic policies |
| Outer iterations | — | 1,001 | Generous budget; convergence typically occurs well before exhaustion |

---

## 🏗 Architecture

```
.
├── 4by4 Gridworld- policy iteration.py          # Standalone script
├── *.ipynb                                       # Interactive Jupyter notebook
├── ARCHITECTURE.md                               # Design rationale & class API
├── CITATION.cff                                  # Academic citation metadata
├── LICENSE                                       # MIT License
└── README.md                                     # ← You are here
```

The core is a single `GridWorld` class encapsulating the full MDP:

```python
class GridWorld:
    actionSpace   # Available actions: {'U', 'D', 'L', 'R'}
    actions       # Valid actions per state (boundary-aware)
    rewards       # Reward signal per state

    getRandomPolicy()   # Initialize π₀
    chooseAction()      # ε-greedy action selection
    greedyChoose()      # argmax over successor values
    move()              # Execute transition, return (s', r)
    getNewState()       # Deterministic dynamics T(s, a) → s'
```

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full API reference and design rationale.

---

## 📊 Convergence Results

The policy evolves from random to optimal. Below is the progression at key checkpoints:

### Initial Random Policy (Step 0)
```
 | R |  | L |  | L |  | D |
 | U |  | U |  | D |  | D |
 | U |  | D |  | D |  | D |
 | R |  | R |  | R |
```
*Chaotic, non-directional — agent wanders aimlessly.*

### Converged Optimal Policy (Step 1000)
```
 | D |  | D |  | D |  | D |
 | R |  | R |  | D |  | D |
 | R |  | D |  | D |  | D |
 | R |  | R |  | R |
```
*Every action points toward the terminal state (3,3) via the shortest safe path, skirting all penalty cells.*

### Exploration vs. Exploitation

```
Exploited: ~78%  │████████████████████████████████████████░░░░░░░░░░░│
Explored:  ~22%  │███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
```

The 5% ε-greedy rate produced roughly a 78/22 exploit/explore split — enough stochasticity to escape local optima while predominantly following the learned policy.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NumPy

### Run

```bash
# Clone
git clone https://github.com/AliGhorbani97/Reinforcement-Learning-solving-a-simple-4by4-Gridworld-using-policy-iteration-in-python.git
cd Reinforcement-Learning-solving-a-simple-4by4-Gridworld-using-policy-iteration-in-python

# Install dependency
pip install numpy

# Execute
python "4by4 Gridworld- policy iteration.py"
```

Or open the Jupyter notebook for an interactive, cell-by-cell walkthrough:

```bash
jupyter notebook Reinforcement_Learning_solving_a_simple_4by4_Gridworld_using_policy_iteration_method.ipynb
```

### Customize the Environment

Modify the reward structure in the `GridWorld.__init__` method:

```python
# Default rewards
self.rewards = {(3, 3): 0.03, (1, 3): -0.01, (2, 1): -0.01, (3, 1): -0.01}

# Example: harsher penalties, bigger goal reward
self.rewards = {(3, 3): 5.0, (1, 3): -2.0, (2, 1): -2.0, (3, 1): -10.0}
```

---

## 📐 Theoretical Foundations

### Bellman Optimality Equation

The value function under the optimal policy satisfies:

$$V^*(s) = \max_{a \in \mathcal{A}(s)} \left[ R(s, a) + \gamma \sum_{s'} P(s' | s, a) \, V^*(s') \right]$$

In our deterministic environment, the transition probability $P(s'|s,a) = 1$ for a single successor, simplifying to:

$$V^*(s) = \max_{a} \left[ R(s, a) + \gamma \cdot V^*(T(s, a)) \right]$$

### Policy Iteration Guarantee

**Theorem (Policy Improvement):** For any policy $\pi$, the greedy policy $\pi'$ defined by

$$\pi'(s) = \arg\max_{a} \left[ R(s,a) + \gamma \cdot V^\pi(T(s,a)) \right]$$

satisfies $V^{\pi'}(s) \geq V^{\pi}(s)$ for all states $s$. Equality holds if and only if $\pi$ is already optimal.

This monotonic improvement guarantee means policy iteration converges in at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations — for our 4×4 grid, this is bounded above by $4^{15}$ but practically converges in far fewer steps.

### Relation to Modern RL

| Concept in This Project | Modern Equivalent |
|------------------------|-------------------|
| Policy Evaluation | Critic network training in Actor-Critic |
| Policy Improvement | Actor update via policy gradient |
| ε-greedy exploration | Entropy regularization in SAC/PPO |
| Value table | Neural network function approximator |
| Gridworld MDP | Simulated environment (MuJoCo, Atari) |

---

## 🔧 Extending This Work

| Extension | Difficulty | Description |
|-----------|:----------:|-------------|
| Stochastic transitions | ⭐ | Add slip probability to `move()` — actions succeed with probability *p*, random otherwise |
| Larger grids | ⭐ | Expand `actions` dictionary to 8×8, 16×16 |
| Value function visualization | ⭐⭐ | Plot heatmaps of V(s) at each iteration using Matplotlib |
| Neural value approximator | ⭐⭐⭐ | Replace the value table with a neural network — bridge to Deep RL |
| Comparison with Q-learning | ⭐⭐ | Implement model-free Q-learning on the same environment and compare sample efficiency |

---

## 📖 Citation

If you use this implementation in academic work, please cite:

```bibtex
@software{ghorbani2026policyiteration,
  author       = {Ghorbani, A.},
  title        = {Policy Iteration on a 4x4 Gridworld -- A Reference Implementation},
  year         = {2026},
  url          = {https://github.com/AliGhorbani97/Reinforcement-Learning-solving-a-simple-4by4-Gridworld-using-policy-iteration-in-python},
  license      = {MIT}
}
```

---

## 📜 License

Released under the [MIT License](LICENSE). Free for academic and commercial use.

---

<div align="center">

*Crafted with rigor by **A. Ghorbani** — advancing the science of intelligent decision-making.*

**Google · Office of the Chief AI Officer**

</div>
