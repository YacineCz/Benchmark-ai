# search_algs.py
# Algorithmes généraux : BFS, DFS, ID, A*, IDA*
# Fournit des wrappers pour mesurer temps/mémoire/nœuds.

import time
import tracemalloc
import collections
import heapq
import math
from typing import Callable, Any, Tuple, List, Set, Dict

class SearchResult:
    def __init__(self, path, cost, nodes_expanded, nodes_generated, time_s, mem_kb):
        self.path = path
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.nodes_generated = nodes_generated
        self.time_s = time_s
        self.mem_kb = mem_kb

def measure_run(fn: Callable, *args, **kwargs) -> SearchResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # fn should return tuple (path, cost, nodes_expanded, nodes_generated)
    path, cost, expanded, generated = res
    return SearchResult(path, cost, expanded, generated, t1 - t0, peak / 1024.0)

# BFS
def bfs(start, goal_test: Callable[[Any], bool], successors: Callable[[Any], List[Tuple[Any,str]]]):
    frontier = collections.deque([start])
    parent = {start: (None, None)}  # state -> (parent, action)
    expanded = 0
    generated = 1
    if goal_test(start):
        return _reconstruct(start, parent), 0, expanded, generated
    while frontier:
        node = frontier.popleft()
        expanded += 1
        for (child, action) in successors(node):
            if child not in parent:
                parent[child] = (node, action)
                generated += 1
                if goal_test(child):
                    return _reconstruct(child, parent), _path_cost(_reconstruct(child, parent)), expanded, generated
                frontier.append(child)
    return None, math.inf, expanded, generated

# DFS (non-recursive, depth limit optional)
def dfs(start, goal_test, successors, depth_limit=None):
    stack = [(start, 0)]
    parent = {start:(None,None)}
    expanded = 0
    generated = 1
    while stack:
        node, d = stack.pop()
        if goal_test(node):
            return _reconstruct(node,parent), _path_cost(_reconstruct(node,parent)), expanded, generated
        if depth_limit is not None and d >= depth_limit:
            continue
        expanded += 1
        for (child, action) in reversed(successors(node)):
            if child not in parent:
                parent[child] = (node, action)
                generated += 1
                stack.append((child, d+1))
    return None, math.inf, expanded, generated

# Iterative Deepening (ID)
def iterative_deepening(start, goal_test, successors, max_depth=50):
    for depth in range(max_depth+1):
        res = dfs(start, goal_test, successors, depth_limit=depth)
        if res[0] is not None:
            return res
    return None, math.inf, None, None

def _reconstruct(state, parent: Dict):
    path = []
    cur = state
    while parent[cur][0] is not None:
        cur, action = parent[cur]
    # We need to rebuild properly: traverse from goal back to start
    # Reconstruct with a two-step method
    rev = []
    cur = state
    while parent[cur][0] is not None:
        p, act = parent[cur]
        rev.append((cur, act))
        cur = p
    rev.reverse()
    actions = [act for (_, act) in rev]
    return actions

def _path_cost(actions):
    return len(actions)

# A* generic
def astar(start, goal_test, successors, heuristic: Callable[[Any], float]):
    open_heap = []
    g = {start: 0}
    f = {start: heuristic(start)}
    parent = {start:(None,None)}
    heapq.heappush(open_heap, (f[start], start))
    closed: Set[Any] = set()
    expanded = 0
    generated = 1
    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if goal_test(node):
            return _reconstruct(node, parent), g[node], expanded, generated
        closed.add(node)
        expanded += 1
        for (child, action, cost) in successors(node):
            tentative_g = g[node] + cost
            if child in closed and tentative_g >= g.get(child, math.inf):
                continue
            if tentative_g < g.get(child, math.inf):
                parent[child] = (node, action)
                g[child] = tentative_g
                fchild = tentative_g + heuristic(child)
                heapq.heappush(open_heap, (fchild, child))
                generated += 1
    return None, math.inf, expanded, generated

# IDA* generic (uses successors returning (child, action, cost))
def ida_star(start, goal_test, successors, heuristic: Callable[[Any], float], max_depth=1000):
    bound = heuristic(start)
    path = [(start, None, 0)]  # (state, action_from_parent, cost_from_parent)
    nodes_generated = 1
    nodes_expanded = 0

    def search(g, bound):
        nonlocal nodes_generated, nodes_expanded
        node = path[-1][0]
        f = g + heuristic(node)
        if f > bound:
            return f
        if goal_test(node):
            actions = []
            for s, a, c in path[1:]:
                actions.append(a)
            return ("FOUND", actions, g, nodes_expanded, nodes_generated)
        min_threshold = math.inf
        nodes_expanded += 1
        for (child, action, cost) in successors(node):
            if any(child == p[0] for p in path):  # avoid cycles
                continue
            path.append((child, action, cost))
            nodes_generated += 1
            t = search(g + cost, bound)
            if isinstance(t, tuple) and t[0] == "FOUND":
                return t
            if t < min_threshold:
                min_threshold = t
            path.pop()
        return min_threshold

    while True:
        t = search(0, bound)
        if isinstance(t, tuple) and t[0] == "FOUND":
            _, actions, cost, nodes_expanded, nodes_generated = t
            return actions, cost, nodes_expanded, nodes_generated
        if t == math.inf or t > max_depth:
            return None, math.inf, nodes_expanded, nodes_generated
        bound = t
