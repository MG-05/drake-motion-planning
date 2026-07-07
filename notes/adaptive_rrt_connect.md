# Adaptive RRT-Connect

This project now uses an adaptive variant of `RRT-Connect` in
[`src/planning/rrt_connect.py`](/Users/mayandgulati/src/UCSB/RoboticsLab/drake-motion-planning/src/planning/rrt_connect.py).
The adaptive logic is enabled through `AdaptiveStepConfig` and is wired into the
task planner and grasp planner option dataclasses.

The implementation is intentionally conservative:

- keep the existing `RRT-Connect` structure
- adapt only the local steering step size
- use the existing Drake collision checker as the source of local geometric
  information
- fall back safely to fixed-step behavior when no clearance estimate is
  available

This note documents the algorithm, the design choices behind it, and the
meaning of every new variable.

## Goal

The original planner used a fixed `step_size` for every expansion:

- small `step_size` works better in the shelf and other narrow passages
- large `step_size` works better in open space and for drop motions

The adaptive variant tries to get both behaviors in one planner:

- use smaller steps near clutter
- use larger steps in open space
- if a proposed extension collides, retry the same target with smaller steps
  before giving up

## High-Level Algorithm

The outer `RRT-Connect` loop is unchanged.

For each expansion:

1. Find the nearest node `q_near`
2. If the node already has a cached clearance estimate, use it
3. Otherwise, first try an aggressive nominal step
4. If that aggressive step collides, estimate local clearance at `q_near`
5. Convert that clearance into a local step size `eta_local`
6. Multiply `eta_local` by a per-tree scale factor that reflects recent success
   or failure
7. Try to extend toward the target using that step size
8. If the edge collides, shrink the step size by `backoff_factor` and retry
9. If all retries fail, mark the extension as `TRAPPED`

The planner still returns the same kind of waypoint path, and shortcutting is
still applied after success when enabled.

## Core Formula

Once a node has a clearance estimate, the local adaptive step size is:

```text
clearance_margin(q_near) = max(clearance(q_near) - minimum_clearance, 0)

eta_base(q_near) = min_step_size + clearance_gain * clearance_margin(q_near)

eta_local(q_near) =
    clip(eta_base(q_near) * tree.step_scale, min_step_size, max_step_size)
```

If no cached clearance is available yet, the planner first tries a nominal
aggressive step. If that succeeds, no clearance query is needed. If the step
fails, the planner computes and caches clearance for that node and switches to
the adaptive formula above.

If no clearance estimator is available at all, the planner falls back to:

```text
eta_local = max_step_size
```

where `max_step_size` defaults to the original planner `step_size`.

## Collision-Triggered Step Backoff

If the first extension attempt fails, the planner does not immediately discard
the sample. It retries the same target with progressively smaller step sizes:

```text
eta_attempt_0 = eta_local
eta_attempt_1 = eta_attempt_0 * backoff_factor
eta_attempt_2 = eta_attempt_1 * backoff_factor
...
```

This is capped by `max_backoff_trials`, and the step size never shrinks below
`min_step_size`.

This design helps in exactly the case that motivated the change:

- a large step works well in open space
- the same step may be too aggressive inside the shelf
- a smaller retry can still make forward progress without forcing the entire
  planner to use a globally small step size

## Per-Tree Online Adaptation

Each tree carries a persistent scalar:

```text
tree.step_scale
```

This encodes a small amount of online experience:

- after a successful extension:

```text
tree.step_scale *= success_growth_factor
```

- after a trapped extension:

```text
tree.step_scale *= failure_shrink_factor
```

The scale is clipped into:

```text
[tree_scale_min, tree_scale_max]
```

This gives each tree a mild tendency to:

- become more aggressive when recent expansions are working
- become more cautious when recent expansions are failing

The effect is intentionally bounded. The goal is local refinement, not a
radically different planner.

## Clearance Source

The collision checker in
[`src/planning/collision.py`](/Users/mayandgulati/src/UCSB/RoboticsLab/drake-motion-planning/src/planning/collision.py)
now exposes:

- `is_free(q)`: boolean feasibility, unchanged
- `is_free.estimate_clearance(q)`: estimated robot-environment signed clearance
- `is_free.minimum_clearance`: the configured feasibility threshold
- `is_free.pair_range`: the signed-distance query horizon

### How clearance is computed

For a configuration `q`:

1. If there is a robot-environment penetration, return the negative penetration
   depth
2. Otherwise, compute the minimum signed distance among robot-environment pairs
   within `pair_range`
3. If no pair is found, return `pair_range` as a lower bound on free-space
   clearance

The estimate is cached per tree node. This means:

- open-space successful expansions avoid unnecessary signed-distance queries
- cluttered nodes still get geometric adaptation once they prove they need it

The adaptive step-size controller still uses the same Drake geometry data as the
feasibility checker, which avoids mismatched notions of “near obstacle”.

## Variables

### Existing planner variables

- `step_size`
  - The nominal maximum extension length in joint space
  - Units: radians in configuration space
  - In the adaptive planner this becomes the default upper bound for local step
    sizing

- `edge_resolution`
  - Interpolation spacing for collision checks along an edge
  - Units: radians in `L_inf` joint-space spacing
  - Smaller values make collision checking denser and slower

- `goal_sample_rate`
  - Probability of directly sampling the goal
  - Higher values bias the planner more strongly toward the goal

- `line_sample_rate`
  - Probability of sampling near the start-goal line with Gaussian noise
  - Helps the planner exploit a plausible corridor without becoming fully greedy

- `sigma_line`
  - Standard deviation of the line-sampling noise
  - Units: joint-space radians

### New adaptive variables

- `enabled`
  - Turns adaptive step sizing on or off
  - If `False`, the planner behaves like the original fixed-step planner

- `min_step_size`
  - Smallest allowed local step size
  - Units: joint-space radians
  - Prevents the planner from shrinking to effectively zero motion

- `max_step_size`
  - Largest allowed local step size
  - Units: joint-space radians
  - If `None`, the planner uses the function argument `step_size`

- `clearance_gain`
  - Maps clearance margin in meters to added joint-space step size in radians
  - Units: radians per meter
  - Larger values make the planner more aggressive as soon as clearance opens up

- `max_backoff_trials`
  - Number of retries after the first failed edge attempt
  - Total attempts per extension are `max_backoff_trials + 1`

- `backoff_factor`
  - Multiplicative shrink applied between retries
  - A value of `0.5` halves the step after each collision

- `success_growth_factor`
  - Multiplier applied to `tree.step_scale` after successful extensions
  - Slightly increases aggressiveness when the tree is progressing well

- `failure_shrink_factor`
  - Multiplier applied to `tree.step_scale` after trapped extensions
  - Slightly decreases aggressiveness when the tree repeatedly fails

- `tree_scale_min`
  - Lower bound on `tree.step_scale`
  - Keeps the online adaptation from collapsing into tiny ineffective steps

- `tree_scale_max`
  - Upper bound on `tree.step_scale`
  - Prevents recent successes from making the tree permanently over-aggressive

## Default Configurations In This Repo

### Task planner transit configuration

Defined in
[`src/manipulation/manipulation_fsm.py`](/Users/mayandgulati/src/UCSB/RoboticsLab/drake-motion-planning/src/manipulation/manipulation_fsm.py):

- `min_step_size = 0.05`
- `clearance_gain = 15.0`
- `max_backoff_trials = 4`
- `backoff_factor = 0.5`
- `success_growth_factor = 1.05`
- `failure_shrink_factor = 0.7`
- `tree_scale_min = 0.5`
- `tree_scale_max = 2.0`

Rationale:

- transit motions should remain reasonably assertive in free space
- shelf-adjacent configurations still need a small enough minimum step to
  squeeze through

### Grasp / retreat configuration

Defined in
[`src/manipulation/grasp.py`](/Users/mayandgulati/src/UCSB/RoboticsLab/drake-motion-planning/src/manipulation/grasp.py):

- `min_step_size = 0.03`
- `clearance_gain = 12.0`
- `max_backoff_trials = 4`
- `backoff_factor = 0.5`
- `success_growth_factor = 1.04`
- `failure_shrink_factor = 0.7`
- `tree_scale_min = 0.5`
- `tree_scale_max = 1.8`

Rationale:

- the grasp-side fallback planner operates closer to contact
- it benefits from a smaller floor and a slightly tighter online scaling range

### Drop / carry configuration

Also defined in
[`src/manipulation/manipulation_fsm.py`](/Users/mayandgulati/src/UCSB/RoboticsLab/drake-motion-planning/src/manipulation/manipulation_fsm.py):

- `min_step_size = 0.08`
- `clearance_gain = 18.0`
- `max_backoff_trials = 4`
- `backoff_factor = 0.5`
- `success_growth_factor = 1.06`
- `failure_shrink_factor = 0.72`
- `tree_scale_min = 0.5`
- `tree_scale_max = 2.3`

Rationale:

- drop motions happen in more open space on average
- the planner should be allowed to recover a larger effective step size

## Why This Design

### Why adapt from clearance at `q_near` instead of sampling a learned policy?

- it uses information already available from Drake
- it is deterministic and easy to debug
- it targets the real problem in this project: narrow shelf regions versus open
  free space

### Why subtract `minimum_clearance`?

The feasibility checker may require a positive clearance margin. A configuration
that is only barely feasible should still be treated as “near obstacle”.

Using:

```text
clearance_margin = clearance - minimum_clearance
```

means the adaptive controller reacts to *extra* room beyond the required safety
buffer, not just absolute distance.

### Why keep the original `RRT-Connect` structure?

- preserves the current planner behavior and code path
- keeps the change local to steering and edge validation
- avoids introducing a second planner that must be maintained separately

### Why retry the same target after collision?

If a large step collides in a tight region, the sample itself may still be
useful. Retrying with a smaller step often salvages the iteration and reduces
repeated sampling waste near the shelf mouth.

### Why use a bounded per-tree scale?

A pure clearance-based controller can still oscillate if local geometry changes
rapidly. The tree-scale state adds a small amount of memory:

- repeated success encourages larger steps
- repeated failure encourages smaller steps

The bounds keep this memory from overpowering the geometry-based component.

## Relationship To Prior Work

This implementation is inspired by, but is not identical to, any single paper.
It combines:

- `RRT-Connect` as the base planner:
  Kuffner and LaValle, 2000
- adaptive step-size ideas for robot-arm planning:
  An, Kim, and Park, 2018
- obstacle-aware local adaptation ideas related to dynamic-domain RRT:
  Yershova et al., 2005 and Jaillet et al., 2005

This codebase deliberately implements a pragmatic hybrid that fits the existing
Drake collision-check interface.

## Future Improvements

- expose planner telemetry: average local step size, number of backoff retries,
  trap rate per phase
- cache or batch clearance queries if Drake query cost becomes significant
- compare against dynamic-domain sampling directly, not just adaptive step size
- add a contextual-bandit selector over a discrete set of step sizes once the
  heuristic adaptive planner has baseline metrics
