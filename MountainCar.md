# Mountain Car — Semi-Gradient SARSA with Tile Coding

An implementation of the classic Mountain Car control problem, using linear
function approximation (tile coding) and semi-gradient SARSA. Built as a
bridge between tabular RL (Gridworld) and full function approximation
(CartPole, Lunar Lander) — the state space here is continuous, so a table
of Q-values isn't an option, but the problem is still small enough to solve
without a neural net.

## The problem

An underpowered car sits in a valley between two hills and needs to reach
a flag at the top of the right hill. The engine alone isn't strong enough
to drive straight up — the agent has to learn to reverse up the left hill
first to build momentum, then use that momentum to escape the valley.

State: `(position, velocity)`, both continuous.
Actions: `reverse`, `none`, `forward`.
Reward: `-1` per step until the goal is reached, so the agent is implicitly
learning to reach the flag in as few steps as possible.

## Why tile coding instead of a neural net

The state space is only 2D, so a full neural function approximator is
overkill — tile coding gives a much simpler, faster-to-train linear
approximator that still generalizes across nearby states.

Tile coding discretizes `(position, velocity)` into a grid of tiles, then
overlays several offset copies of that grid ("tilings"). A given state
activates one tile per tiling — several active features per state instead
of a table lookup — with nearby states sharing some but not all of those
tiles. This is what lets the Q-function generalize smoothly instead of
treating every state independently.

**Implementation specifics:**
- 8 tilings, 8×8 tiles per tiling
- Each tiling is offset from the others using strides of 3 (position) and
  5 (velocity) — both coprime with the number of tilings (8), so no two
  tilings end up with the same offset
- Total feature vector length: `num_tilings * tiles_per_dim^2 * num_actions`,
  and since features are binary, `Q(s,a)` is just a sum over the active
  tile weights rather than a full dot product

## Why SARSA over Q-learning

SARSA is on-policy: it updates toward the value of the action the policy
actually takes next, rather than the greedy max like Q-learning does. With
function approximation, on-policy updates are more stable — the "semi" in
semi-gradient means the update ignores how the target itself depends on
the weights (since Q is linear in `w`, the gradient is just the feature
vector, which keeps the update cheap and well-behaved).

This implementation also takes a shortcut for the update step: rather than
computing the gradient over the full feature vector, it updates `w` only
at the indices returned by `get_active_tiles`, since every other entry in
the gradient is zero anyway.

## Files

- `mountain_car.py` — environment, tile coder, linear Q-function,
  epsilon-greedy policy, and the SARSA training loop

## Running it

```bash
python mountain_car.py
```

Trains for 2500 episodes with epsilon decaying by 0.99 per episode, then
plots the episode-length learning curve and the learned cost-to-go surface
over `(position, velocity)`.

## What I'd try next

- Compare against a Q-learning (off-policy) version on the same tile coder
- Try eligibility traces (SARSA(λ)) to speed up early learning
- Sweep tile resolution / number of tilings to see the generalization vs.
  precision trade-off directly
