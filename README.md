# Matching Pennies - R-NaD

Two RL agents learn to play the zero-sum game **Matching Pennies** using **Regularized Nash Dynamics (R-NaD)**, converging to the Nash equilibrium (50/50 mixed strategy).

## The Game

- Player A and Player B simultaneously choose Heads or Tails
- **Same** = Player A wins (+1 for A, -1 for B)
- **Different** = Player B wins (-1 for A, +1 for B)
- **Nash equilibrium**: both players play 50/50

## Approach

Each player is a small MLP that outputs a probability distribution over {Heads, Tails}. Both agents are trained with REINFORCE (policy gradient).

### The Problem: Cycling (Vanilla REINFORCE)

With vanilla REINFORCE, the agents cycle endlessly -- policies slam between 0 and 1, NashConv spikes to ~4.0, and the Lyapunov potential never decays. This is replicator dynamics in action.

![Vanilla REINFORCE - Cycling](convergence_vanilla.png)

### The Fix: R-NaD

R-NaD adds a KL divergence penalty toward a periodically-updated reference policy. This acts as "friction" that dampens oscillations, enabling convergence to Nash.

- **Inner loop**: policy gradient + KL penalty toward frozen reference
- **Outer loop**: periodically update reference to current policy

![R-NaD - Convergence](convergence_rnad.png)

### Results Comparison

| Metric | Vanilla REINFORCE | R-NaD |
|---|---|---|
| P(Heads) range | 0.0 - 1.0 (wild swings) | 0.42 - 0.60 (near Nash) |
| NashConv (exploitability) | Spikes to ~4.0 | Hovers around 0.05 - 0.15 |
| Lyapunov potential | ~3.0 - 4.0 (no decay) | ~0.0 - 0.03 (near zero) |

## Understanding the Plots

Each run produces three plots:

### Policy Convergence (left panel)
Tracks P(Heads) for both players over training steps. The red dashed line at 0.5 is the Nash equilibrium.
- **Vanilla**: Both players swing between deterministic strategies (0.0 and 1.0). When A commits to Heads, B exploits this by switching to Tails, then A adapts, and the cycle repeats. This is a discrete-time manifestation of replicator dynamics, which provably cycles in zero-sum games.
- **R-NaD**: Both players stay close to 0.5 with small fluctuations. The KL penalty toward the reference policy acts as a leash, preventing either agent from overcommitting to a pure strategy. The remaining noise comes from REINFORCE variance (single-sample reward estimates).

### NashConv / Exploitability (middle panel)
Measures how much each player could gain by switching to a best response against the opponent's current strategy, summed over both players. At Nash equilibrium this is 0 -- neither player benefits from deviating.
- **Vanilla**: Exploitability spikes whenever a player commits to a near-pure strategy, since the opponent's best response would yield a large payoff. The spikes never shrink because the system has no mechanism to dampen oscillations.
- **R-NaD**: Exploitability stays low because neither player ever strays far enough from 50/50 to be significantly exploitable.

### Lyapunov Potential Decay (right panel)
Tracks KL(pi_A || uniform) + KL(pi_B || uniform) -- the total divergence of both policies from the Nash equilibrium. This serves as a Lyapunov-style energy function: if it trends toward zero, the system is converging.
- **Vanilla**: The potential oscillates at high values (3.0-4.0) indefinitely. There is no "friction" in the dynamics, so energy is conserved and the system orbits the equilibrium without approaching it.
- **R-NaD**: The potential stays near zero (0.0-0.03). The KL regularization introduces friction that dissipates energy, causing the system to spiral inward toward equilibrium rather than orbit around it.

## Run

```bash
python training-vanilla.py   # Vanilla REINFORCE (cycling)
python training-RNaD.py      # R-NaD (convergence)
```
