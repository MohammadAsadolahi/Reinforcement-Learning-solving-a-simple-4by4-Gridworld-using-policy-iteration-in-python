# Architecture & Design Rationale

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   Policy Iteration                  │
│                   (Outer Loop)                      │
│                                                     │
│   ┌─────────────────┐     ┌─────────────────────┐   │
│   │  Policy          │     │  Policy              │   │
│   │  Evaluation      │────▶│  Improvement          │   │
│   │  (Value Update)  │     │  (Greedy Selection)  │   │
│   └────────┬────────┘     └──────────┬──────────┘   │
│            │                         │               │
│            └─────────────────────────┘               │
│                    Converged?                        │
│                    Yes → Stop                        │
└─────────────────────────────────────────────────────┘
```

## Class: `GridWorld`

| Method | Purpose |
|---|---|
| `__init__` | Defines state space, action space, transition dynamics, and reward structure |
| `getRandomPolicy` | Initializes a stochastic baseline policy |
| `reset` | Returns agent to start state `(0, 0)` |
| `is_terminal` | Checks if a state is absorbing |
| `getNewState` | Deterministic transition function $s' = T(s, a)$ |
| `chooseAction` | ε-greedy action selection for exploration–exploitation trade-off |
| `greedyChoose` | Selects the action maximizing successor state value |
| `move` | Executes action, returns `(next_state, reward)` |
| `printValues` | Renders the value table as a grid |
| `printPolicy` | Renders the policy map as a grid |

## Convergence Dynamics

The algorithm alternates between **evaluation** (1,000 episodes per iteration) and **improvement** (greedy policy update over all states). With a discount factor γ = 0.1 and exploration rate ε = 0.05, the policy typically converges within 400–600 outer iterations.
