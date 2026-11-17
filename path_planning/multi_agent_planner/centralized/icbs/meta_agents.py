"""Meta-agents for ICBS: one joint plan for a group of agents that keep conflicting. Mixed into
``icbs.IEnvironment``."""
import heapq
import time
from itertools import count

from path_planning.multi_agent_planner.centralized.cbs.data_types import (
    Constraints, EdgeConstraint, Location, State, VertexConstraint, _turn_toward)


def _balls_collide(p, q, u, v, reach: float) -> bool:
    """Whether two balls whose radii sum to ``reach`` overlap while their centres go ``p -> q`` and
    ``u -> v`` over the same unit step (straight, at constant speed; a wait is ``p == q``). Touching is
    not a collision."""
    d = [a - b for a, b in zip(p, u)]                            # relative position at the start
    w = [(b - a) - (y - x) for a, b, x, y in zip(p, q, u, v)]    # relative velocity
    ww = sum(x * x for x in w)
    s = 0.0 if ww == 0.0 else min(1.0, max(0.0, -sum(x * y for x, y in zip(d, w)) / ww))
    return sum((x + s * y) ** 2 for x, y in zip(d, w)) < (reach - 1e-9) ** 2


class JointSearch:
    """The joint low-level search of ``IEnvironment``. Uses its ``agent_dict``, ``agent_type``,
    ``turn_rate``, ``has_bodies``, ``admissible_heuristic``, ``_moves_from`` and ``graph_map``."""

    def joint_search(self, group, constraint_dict, max_expansions: int = 200000, stop_at: float | None = None):
        """Optimal joint plan of the agents ``group`` (a meta-agent): A* over their joint positions in unit
        steps. Point agents never share a vertex or swap along an edge; discs / spheres never overlap
        during a step, move only the way they face and spend a step to turn. Each agent obeys its own
        constraints. An agent's cost runs until it stops at its
        goal for good (it may leave and come back before that). Returns ``({agent: [State, ...]},
        {agent: cost})``, or False when no plan is found within ``max_expansions`` or before the clock
        time ``stop_at`` (the caller's time limit: a joint search can be long)."""
        agents = list(group)
        k = len(agents)
        cons = [constraint_dict.get(a) or Constraints() for a in agents]
        starts = tuple((s.location.point, s.ori) for s in (self.agent_dict[a]["start"] for a in agents))   # (point, heading)
        goals = [self.agent_dict[a]["goal"].location.point for a in agents]
        t0 = self.agent_dict[agents[0]]["start"].time
        # The first time each agent may stop for good: after every constraint that concerns it later.
        free_from = [con.free_from(Location(goal)) for con, goal in zip(cons, goals)]
        h = lambda i, pos: self.admissible_heuristic(State(0, Location(pos[0]), pos[1]), agents[i])   # pos: (point, heading)
        locations = {}

        def loc(p):
            cached = locations.get(p)
            if cached is None:
                cached = locations[p] = Location(p)
            return cached

        def allowed(con, t, p, q):
            if con.deadline is not None and t + 1 > con.deadline:
                return False
            if VertexConstraint(t + 1, loc(q)) in con.vertex_constraints:
                return False
            if EdgeConstraint(t, loc(p), loc(q)) in con.edge_constraints:
                return False
            for begin, vertices, edges in con.regions:
                if (t + 1 >= begin and loc(q) in vertices) or (t >= begin and (loc(p), loc(q)) in edges):
                    return False
            need = con.must_be.get(t + 1)
            return need is None or need.point == q

        def may_stop(i, t):                                      # at its goal, for good
            con = cons[i]
            return t >= free_from[i] and (con.deadline is None or t <= con.deadline) \
                and not any(loc(goals[i]) in vertices for _, vertices, _ in con.regions)

        radii = [self.agent_type(a).reach for a in agents] if self.has_bodies else None
        rates = [self.turn_rate(a) for a in agents]              # degrees per step; None: instant turns

        def clash(i, p, q, j, u, v):                             # agents i and j over one step
            if radii is None:
                return q == v or (q == u and v == p and q != v)   # same vertex, or a swap
            return _balls_collide(p, q, u, v, radii[i] + radii[j])

        def joint_steps(options, locs, chosen=()):
            """Every combination of one option per agent in which no two of them collide."""
            i = len(chosen)
            if i == k:
                yield chosen
                return
            for option in options[i]:
                if not any(clash(j, locs[j][0], chosen[j][0][0], i, locs[i][0], option[0][0]) for j in range(i)):
                    yield from joint_steps(options, locs, chosen + (option,))

        start = (t0, starts, (False,) * k)
        best_g, parent, closed, tick = {start: 0.0}, {}, set(), count()
        heap = [(sum(h(i, pos) for i, pos in enumerate(starts)), 0.0, next(tick), start)]
        while heap and len(closed) < max_expansions and (stop_at is None or time.time() < stop_at):
            _, _, _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            t, locs, done = cur
            if all(done):
                break
            options = []
            for i in range(k):                                   # an option: ((point, heading), stopped, cost)
                p, ori = locs[i]
                if done[i]:
                    options.append([(locs[i], True, 0.0)])
                    continue
                opts = [(locs[i], True, 0.0)] if p == goals[i] and may_stop(i, t) else []
                stay = allowed(cons[i], t, p, p)
                if stay:
                    opts.append((locs[i], False, 1.0))
                turns = {}                                       # an ordered set of the headings one turn step away
                for q, _, facing in self._moves_from(p):
                    if ori is None or ori == facing:                     # it moves only the way it faces
                        if allowed(cons[i], t, p, q):
                            opts.append(((q, None if ori is None else facing), False,
                                         float(self.graph_map.get_distance(p, q))))
                    else:
                        turns.setdefault(_turn_toward(ori, facing, rates[i]))
                if stay and turns and EdgeConstraint(t, loc(p), loc(p), True) not in cons[i].edge_constraints:
                    opts += [((p, heading), False, 1.0) for heading in turns]    # one step of a turn in place
                options.append(opts)
            for combo in joint_steps(options, locs):
                new = tuple(o[0] for o in combo)
                nxt = (t + 1, new, tuple(o[1] for o in combo))
                if nxt in closed:
                    continue
                g = best_g[cur] + sum(o[2] for o in combo)
                if g < best_g.get(nxt, float("inf")) - 1e-12:
                    best_g[nxt] = g
                    parent[nxt] = (cur, tuple(o[2] for o in combo))
                    f = g + sum(0.0 if nxt[2][i] else h(i, new[i]) for i in range(k))
                    heapq.heappush(heap, (f, -g, next(tick), nxt))
        else:
            return False
        states, costs = [cur], [0.0] * k
        while states[-1] in parent:
            prev, step = parent[states[-1]]
            costs = [c + d for c, d in zip(costs, step)]
            states.append(prev)
        states.reverse()
        paths = {}
        for i, a in enumerate(agents):                          # each path ends when its agent stops for good
            paths[a] = [State(t, loc(locs[i][0]), locs[i][1]) for t, locs, done in states if not done[i]]
        return paths, dict(zip(agents, costs))
