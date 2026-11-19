"""The SIPP low-level search for agents with bodies, compiled with Numba.

It does what ``SippPlanner``'s Python search does, step for step and with the same tie-breaking, on array
copies of the safe-interval table (``SippGraph.arrays``):

- states are (vertex, safe interval), keyed by the vertex and the interval's two ends, so two interval
  lists at one vertex with the same interval give one state, as in the Python dict keys;
- the open list is ordered by (f, push counter), as Python's ``heapq`` of ``(f, counter, state)``;
- successors are generated in the same order (moves, then the next vertex's intervals, then the edge's
  departure windows), with the same arithmetic, including ``SippNode.split_interval``'s 1e-10 buffer and
  1e-6 minimum window;
- an agent that is not a ball prunes redundant arrivals (no later, may stay no longer) as Python does.

The search runs in chunks and can stop part-way: when the next state to expand has a neighbour whose
per-heading intervals are not worked out yet (``NEED``), when it needs more room (``GROW``) or after a
chunk of expansions (``CHUNK``, so the caller can check its time limit). All its state lives in the arrays
passed in, so calling it again carries on where it stopped.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

FOUND, EXHAUSTED, CAPPED, NEED, GROW, CHUNK = 0, 1, 2, 3, 4, 5
INF = np.inf

KEY = types.Tuple((types.int64, types.float64, types.float64))


@njit(cache=True)
def new_index(v, lo, hi):
    """The (vertex, lo, hi) -> state id dict the search keeps between calls, holding the start state (id 0).
    Made in compiled code so that it compiles once into Numba's disk cache, not in every new process."""
    index = Dict.empty(key_type=KEY, value_type=types.int64)
    index[(v, lo, hi)] = 0
    return index


@njit(cache=True)
def _heap_less(f, c, i, j):
    return f[i] < f[j] or (f[i] == f[j] and c[i] < c[j])


@njit(cache=True)
def _heap_push(hf, hc, hs, ht, meta, f, cnt, sid, t):
    n = meta[1]
    hf[n], hc[n], hs[n], ht[n] = f, cnt, sid, t
    meta[1] = n + 1
    while n > 0:
        p = (n - 1) // 2
        if _heap_less(hf, hc, n, p):
            hf[n], hf[p] = hf[p], hf[n]
            hc[n], hc[p] = hc[p], hc[n]
            hs[n], hs[p] = hs[p], hs[n]
            ht[n], ht[p] = ht[p], ht[n]
            n = p
        else:
            break


@njit(cache=True)
def _heap_pop(hf, hc, hs, ht, meta):
    n = meta[1] - 1
    meta[1] = n
    hf[0], hc[0], hs[0], ht[0] = hf[n], hc[n], hs[n], ht[n]
    i = 0
    while True:
        l, r, m = 2 * i + 1, 2 * i + 2, i
        if l < n and _heap_less(hf, hc, l, m):
            m = l
        if r < n and _heap_less(hf, hc, r, m):
            m = r
        if m == i:
            break
        hf[i], hf[m] = hf[m], hf[i]
        hc[i], hc[m] = hc[m], hc[i]
        hs[i], hs[m] = hs[m], hs[i]
        ht[i], ht[m] = ht[m], ht[i]
        i = m


@njit(cache=True)
def _bisect_left(a, n, x):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bisect_right(a, n, x):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def _departures(now, leave_by, m_time, e_lo, e_hi, e_n, out_lo, out_hi, tmp_lo, tmp_hi):
    """When the agent may start along the edge: ``[(now, leave_by)]`` less ``[t1 - m_time, t2]`` for each
    unsafe window overlapping ``[now, leave_by + m_time]``, split as ``SippNode.split_interval`` does, then
    the windows reaching into ``[now, leave_by]``. Returns how many were written. ``tmp_lo`` / ``tmp_hi``:
    scratch space."""
    end_t = leave_by + m_time
    k0 = _bisect_left(e_hi, e_n, now)                 # first unsafe window ending at or after now
    k1 = _bisect_right(e_lo, e_n, end_t)              # first starting after end_t
    out_lo[0], out_hi[0] = now, leave_by
    if k0 >= k1:
        return 1                                       # nothing unsafe: the whole window, as is
    cnt = 1
    for k in range(k0, k1):
        b_start = (e_lo[k] - m_time) - 1e-10
        b_end = e_hi[k] + 1e-10
        m = 0
        for j in range(cnt):
            s, e = out_lo[j], out_hi[j]
            if b_start >= e or b_end <= s:
                tmp_lo[m], tmp_hi[m] = s, e
                m += 1
            else:
                if b_start > s:
                    tmp_lo[m], tmp_hi[m] = s, b_start
                    m += 1
                if b_end < e:
                    tmp_lo[m], tmp_hi[m] = b_end, e
                    m += 1
        cnt = 0
        for j in range(m):                     # drop slivers, keep sorted (insertion sort)
            if tmp_hi[j] - tmp_lo[j] > 1e-6:
                p = cnt
                while p > 0 and (out_lo[p - 1] > tmp_lo[j] or (out_lo[p - 1] == tmp_lo[j] and out_hi[p - 1] > tmp_hi[j])):
                    out_lo[p], out_hi[p] = out_lo[p - 1], out_hi[p - 1]
                    p -= 1
                out_lo[p], out_hi[p] = tmp_lo[j], tmp_hi[j]
                cnt += 1
    keep = 0
    for j in range(cnt):
        if out_lo[j] <= leave_by and out_hi[j] >= now:
            out_lo[keep], out_hi[keep] = out_lo[j], out_hi[j]
            keep += 1
    return keep


@njit(cache=True)
def search_chunk(
        # the roadmap moves for this agent: CSR by source vertex
        mv_ptr, mv_to, mv_dur, mv_edge, mv_node,
        # vertex / per-heading node safe intervals, edge unsafe windows
        nd_lo, nd_hi, nd_n, ed_lo, ed_hi, ed_n,
        # the target and the estimate
        h, goal, require_forever, oriented,
        # search state (kept between calls)
        index, st_v, st_lo, st_hi, g, closed, parent, parent_t, act_w, act_m,
        hf, hc, hs, ht, ar_head, ar_next, ar_g, ar_e, ar_alive, meta,
        max_iterations, chunk, need_out, goal_t):
    """Advance the search by at most ``chunk`` expansions. ``meta``: [states, heap size, push counter,
    iterations, arrival entries, goal state]; ``goal_t``: the goal's arrival time. Returns a status code
    (see the module constants)."""
    win_lo = np.empty(nd_lo.shape[1] * 2 + ed_lo.shape[1] * 2 + 4)
    win_hi = np.empty_like(win_lo)
    tmp_lo = np.empty(2 * ed_lo.shape[1] + 4)            # (_departures' scratch: one window more per unsafe one)
    tmp_hi = np.empty_like(tmp_lo)
    done = 0
    while True:
        if meta[3] >= max_iterations:
            return CAPPED
        if meta[1] == 0:
            return EXHAUSTED
        if done >= chunk:
            return CHUNK
        sid = hs[0]
        if closed[sid]:
            meta[3] += 1
            _heap_pop(hf, hc, hs, ht, meta)
            continue
        u = st_v[sid]
        # Every neighbour's intervals have to be known before expanding (per-heading ones on demand).
        nneed = 0
        for k in range(mv_ptr[u], mv_ptr[u + 1]):
            if mv_node[k] < 0:
                need_out[nneed] = mv_to[k]
                nneed += 1
        if nneed > 0:
            need_out[nneed] = -1
            return NEED
        # Room for every successor this expansion could make.
        bound = (mv_ptr[u + 1] - mv_ptr[u]) * nd_lo.shape[1] * (ed_lo.shape[1] + 1) + 1
        if meta[0] + bound > st_v.shape[0] or meta[1] + bound > hf.shape[0] or meta[4] + bound > ar_g.shape[0]:
            return GROW
        meta[3] += 1
        now = ht[0]
        _heap_pop(hf, hc, hs, ht, meta)
        closed[sid] = True
        done += 1
        g_cur = g[sid]
        leave_by = st_hi[sid]
        for k in range(mv_ptr[u], mv_ptr[u + 1]):
            w, m_time, node, e = mv_to[k], mv_dur[k], mv_node[k], mv_edge[k]
            start_t = now + m_time
            end_t = leave_by + m_time
            nint = nd_n[node]
            any_iv = False
            for j in range(nint):
                if nd_lo[node, j] <= end_t and nd_hi[node, j] >= start_t:
                    any_iv = True
                    break
            if not any_iv:
                continue
            if e >= 0:
                nwin = _departures(now, leave_by, m_time, ed_lo[e], ed_hi[e], ed_n[e], win_lo, win_hi, tmp_lo, tmp_hi)
            else:
                win_lo[0], win_hi[0] = now, leave_by
                nwin = 1
            for j in range(nint):
                ilo, ihi = nd_lo[node, j], nd_hi[node, j]
                if not (ilo <= end_t and ihi >= start_t):
                    continue
                for q in range(nwin):
                    wlo, whi = win_lo[q], win_hi[q]
                    arrive = start_t
                    if ilo > arrive:
                        arrive = ilo
                    if wlo + m_time > arrive:
                        arrive = wlo + m_time
                    depart = arrive - m_time
                    if not (depart >= wlo - 1e-9 and depart <= whi):
                        continue
                    if not (arrive >= ilo and arrive <= ihi):
                        continue
                    key = (w, ilo, ihi)
                    s2 = index[key] if key in index else np.int64(-1)
                    if s2 >= 0 and closed[s2]:
                        continue
                    cost = arrive - now
                    tent = g_cur + cost
                    if oriented:
                        dominated = False
                        a = ar_head[w]
                        while a >= 0:
                            if ar_alive[a] and ar_g[a] <= tent + 1e-9 and ar_e[a] >= ihi - 1e-9:
                                dominated = True
                                break
                            a = ar_next[a]
                        if dominated:
                            continue
                        a, prev = ar_head[w], -1          # drop what this arrival makes redundant
                        while a >= 0:
                            if ar_alive[a] and tent <= ar_g[a] + 1e-9 and ihi >= ar_e[a] - 1e-9:
                                ar_alive[a] = False
                            if ar_alive[a]:
                                prev = a
                            elif prev < 0:                # (and unlink every dead entry)
                                ar_head[w] = ar_next[a]
                            else:
                                ar_next[prev] = ar_next[a]
                            a = ar_next[a]
                        n_ar = meta[4]
                        ar_g[n_ar], ar_e[n_ar], ar_alive[n_ar], ar_next[n_ar] = tent, ihi, True, ar_head[w]
                        ar_head[w] = n_ar
                        meta[4] = n_ar + 1
                    if s2 < 0:
                        s2 = meta[0]
                        meta[0] = s2 + 1
                        index[key] = s2
                        st_v[s2], st_lo[s2], st_hi[s2] = w, ilo, ihi
                        g[s2] = INF
                        closed[s2] = False
                    if tent < g[s2]:
                        parent[s2], parent_t[s2] = sid, now
                        g[s2] = tent
                        act_w[s2], act_m[s2] = cost - m_time, m_time
                        if w == goal and (not require_forever or ihi == INF):
                            meta[5] = s2
                            goal_t[0] = arrive
                            return FOUND
                        f = g[s2] + h[w]
                        _heap_push(hf, hc, hs, ht, meta, f, meta[2], s2, arrive)
                        meta[2] += 1
