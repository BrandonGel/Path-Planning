# Neural-ATTF / SIPP planner changes (`mapd` branch)

This change set adds **run metrics**, fixes a **deadlock-recovery loop**, and makes the
decentralized planner **time-unit aware**. Two files are touched:

- `path_planning/multi_agent_planner/centralized/sipp/sipp.py`
- `path_planning/multi_agent_planner/decentralized/neural_attf/neural_attf.py`

---

## 1. SIPP low-level effort metric (`sipp.py`)

`SippPlanner.compute_plan` now accumulates low-level search expansions across all
high-level iterations and reports the total:

```python
solution_info["low_level_iterations"]  # total A*/SIPP expansions for this plan
```

No behavior change — purely instrumentation consumed by the metrics below.

---

## 2. Time-unit awareness (`neural_attf.py`)

SIPP plans in **seconds** (edge time = distance / velocity), but the simulation steps in
discrete ticks. Two new constructor parameters bridge them:

| Param | Default | Meaning |
|-------|---------|---------|
| `sipp_time_limit` | `None` | Wall-clock cap (s) per SIPP low-level call. `None` = unbounded (stops only at the iteration cap, which can be slow on large roadmaps). |
| `timestep_duration` | `1.0` | Real seconds each simulation step represents. A step covers ≈ `timestep_duration * velocity` world units; larger values coarsen stepping (fewer positions per route). |

Effects:

- **`_other_agent_schedules`** stamps other agents' committed paths at SIPP time
  `k * timestep_duration` (step index `k`), so dynamic obstacles align with SIPP's
  continuous clock.
- **`_resample_unit(schedule, goal, dt=...)`** resamples a continuous-time SIPP schedule
  at `dt`-second steps. Waits become repeated positions, long edges are subdivided, and
  the path ends exactly on `goal`.
- **`plan`** passes `time_limit=self.sipp_time_limit` into `SippPlanner`.

> Requires `SippPlanner.__init__` to accept a `time_limit` argument.

---

## 3. Deadlock-recovery rewrite (the core fix)

**Problem:** the old recovery picked a *random* nearby free node, so a stuck agent often
evacuated straight onto another agent's route — producing a repeated-deadlock loop.

**Fix:** recovery is now interference-aware and deterministic.

- `_interference_footprint(agent_name)` — the set of graph-node coords an agent must not
  rest on: every *other* agent's moving + idle footprint (radius-aware), all path ends,
  occupied parking spots, **every waypoint of pending/assigned tasks**, and task
  start/goal cells. The task-waypoint inclusion is what breaks the loop.
- `_close_non_interfering_nodes(agent_pos, agent_name, r)` — reachable nodes within radius
  `r` whose resting footprint is clear, **sorted nearest-first**. Local-only: empty list
  ⇒ the agent stays put. Uses a `scipy.spatial.KDTree` ball query (clearance =
  `2 * agent_radius`) instead of a per-candidate CGAL sweep, which dominated runtime over
  ~1500 nodes per call.
- `deadlock_recovery` — keeps its count-gate, then tries candidates nearest-first and
  commits the **first one whose trajectory also plans collision-free** (SIPP treats other
  agents' paths as dynamic obstacles, so a successful plan = no interference).

`_random_close_node_point` and the `random` import are removed.

---

## 4. Parking back-off

A stuck agent used to re-run SIPP toward a parking endpoint **every step**. Now:

- `go_to_closest_non_task_endpoint` returns `True` (already parked / path committed) or
  `False` (no endpoint path — stuck).
- On failure the agent's parking dispatch is skipped for `park_retry_cooldown` steps
  (default `3`) via `token["park_retry_after"]`. A new task assignment still routes the
  agent through the task branch, so it is never starved.

---

## 5. Metrics bookkeeping & getters

New token fields, with getters:

| Getter | Returns |
|--------|---------|
| `get_assigned_tasks_times()` | task name → time first assigned |
| `get_assigned_tasks_agent()` | task name → owning agent at assignment |
| `get_start_tasks_times()` | task start times |
| `get_sipp_iterations()` | total SIPP low-level expansions across the run |
| `get_sipp_calls()` | number of SIPP low-level calls |
| `get_sipp_iterations_max_seen()` | worst single SIPP call |

SIPP counts are sourced from `solution_info["low_level_iterations"]` (see §1).

---

## Verify

```bash
python -m py_compile \
  path_planning/multi_agent_planner/centralized/sipp/sipp.py \
  path_planning/multi_agent_planner/decentralized/neural_attf/neural_attf.py
```
