"""
Benchmarking script for Mini-projet: Benchmarking et analyse comparative de méthodes de recherche
Supports: BFS, DFS, ID (iterative deepening), UCS, A*, IDA*
Problems: Sliding Puzzle (Taquin), Grid Maze (Shortest Path), Missionaries-Cannibals
Outputs: CSV with metrics, simple matplotlib visualizations

Usage: python benchmark.py

Dependencies: numpy (optional), matplotlib
"""

import time
import tracemalloc
import csv
import heapq
import math
from collections import deque, defaultdict
import itertools
import json
import os

# --------------------------- Utilities for measurement ---------------------------
class Metrics:
    def __init__(self):
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def reset(self):
        self.nodes_expanded = 0
        self.nodes_generated = 0

# helper to measure time and memory of a function
def measure(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    metrics = Metrics()
    result = func(*args, metrics=metrics, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'result': result,
        'time_s': t1 - t0,
        'mem_kb_peak': peak / 1024.0,
        'mem_kb_current': current / 1024.0,
        'nodes_expanded': metrics.nodes_expanded,
        'nodes_generated': metrics.nodes_generated,
    }

# --------------------------- Generic search algorithms ---------------------------

def bfs(start, is_goal, get_neighbors, metrics=None):
    if metrics is None: metrics = Metrics()
    frontier = deque([start])
    came_from = {start: None}
    metrics.nodes_generated += 1
    while frontier:
        node = frontier.popleft()
        metrics.nodes_expanded += 1
        if is_goal(node):
            return reconstruct_path(came_from, node)
        for neighbor in get_neighbors(node):
            metrics.nodes_generated += 1
            if neighbor not in came_from:
                came_from[neighbor] = node
                frontier.append(neighbor)
    return None


def dfs(start, is_goal, get_neighbors, limit=None, metrics=None):
    if metrics is None: metrics = Metrics()
    visited = set()
    stack = [(start, None, 0)]  # (node, parent, depth)
    parents = {start: None}
    metrics.nodes_generated += 1
    while stack:
        node, parent, depth = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        metrics.nodes_expanded += 1
        if is_goal(node):
            return reconstruct_path(parents, node)
        if limit is not None and depth >= limit:
            continue
        for neighbor in get_neighbors(node):
            metrics.nodes_generated += 1
            if neighbor not in visited:
                parents[neighbor] = node
                stack.append((neighbor, node, depth + 1))
    return None


def iterative_deepening(start, is_goal, get_neighbors, max_depth=50, metrics=None):
    if metrics is None: metrics = Metrics()
    for depth in range(max_depth + 1):
        # call DFS with limit and separate metrics per iteration but accumulate
        iter_metrics = Metrics()
        res = dfs(start, is_goal, get_neighbors, limit=depth, metrics=iter_metrics)
        metrics.nodes_expanded += iter_metrics.nodes_expanded
        metrics.nodes_generated += iter_metrics.nodes_generated
        if res is not None:
            return res
    return None


def ucs(start, is_goal, get_neighbors_cost, metrics=None):
    if metrics is None: metrics = Metrics()
    # get_neighbors_cost(node) -> iterable of (neighbor, cost)
    frontier = []
    heapq.heappush(frontier, (0, start))
    cost_so_far = {start: 0}
    came_from = {start: None}
    metrics.nodes_generated += 1
    while frontier:
        cost, node = heapq.heappop(frontier)
        metrics.nodes_expanded += 1
        if is_goal(node):
            return reconstruct_path(came_from, node)
        for neighbor, step_cost in get_neighbors_cost(node):
            metrics.nodes_generated += 1
            new_cost = cost_so_far[node] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = node
    return None


def a_star(start, is_goal, get_neighbors_cost, heuristic, metrics=None):
    if metrics is None: metrics = Metrics()
    frontier = []
    start_h = heuristic(start)
    heapq.heappush(frontier, (start_h, 0, start))  # (f, g, node)
    came_from = {start: None}
    g_score = {start: 0}
    metrics.nodes_generated += 1
    while frontier:
        f, g, node = heapq.heappop(frontier)
        metrics.nodes_expanded += 1
        if is_goal(node):
            return reconstruct_path(came_from, node)
        for neighbor, step_cost in get_neighbors_cost(node):
            metrics.nodes_generated += 1
            tentative_g = g_score[node] + step_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))
                came_from[neighbor] = node
    return None


def ida_star(start, is_goal, get_neighbors_cost, heuristic, metrics=None, max_iterations=100000):
    if metrics is None: metrics = Metrics()
    # Node representation must be hashable
    bound = heuristic(start)
    path = [start]
    g = {start: 0}

    def search(path, g_cost, bound):
        node = path[-1]
        f = g_cost + heuristic(node)
        if f > bound:
            return f, None
        if is_goal(node):
            return 'FOUND', list(path)
        min_threshold = float('inf')
        for neighbor, step_cost in get_neighbors_cost(node):
            metrics.nodes_generated += 1
            if neighbor in path:
                continue
            path.append(neighbor)
            metrics.nodes_expanded += 1
            g_next = g_cost + step_cost
            t, res = search(path, g_next, bound)
            if t == 'FOUND':
                return 'FOUND', res
            if t < min_threshold:
                min_threshold = t
            path.pop()
        return min_threshold, None

    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations:
            return None
        t, res = search(path, 0, bound)
        if t == 'FOUND':
            return res
        if t == float('inf'):
            return None
        bound = t

# --------------------------- Helpers ---------------------------

def reconstruct_path(came_from, node):
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    return path

# --------------------------- Problem 1: Sliding Puzzle (Taquin) ---------------------------
# State: tuple of integers length N*N where 0 represents empty


def sliding_neighbors(state, N):
    zero_idx = state.index(0)
    r, c = divmod(zero_idx, N)
    moves = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < N and 0 <= nc < N:
            ni = nr*N + nc
            new = list(state)
            new[zero_idx], new[ni] = new[ni], new[zero_idx]
            moves.append(tuple(new))
    return moves


def sliding_neighbors_cost(state, N):
    for nb in sliding_neighbors(state, N):
        yield (nb, 1)


def sliding_goal(state, goal):
    return state == goal


def heuristic_misplaced(state, goal):
    return sum(1 for a,b in zip(state, goal) if a != b and a != 0)


def heuristic_manhattan(state, goal, N):
    pos_goal = {val: i for i,val in enumerate(goal)}
    s = 0
    for idx, val in enumerate(state):
        if val == 0: continue
        goal_idx = pos_goal[val]
        r1,c1 = divmod(idx, N)
        r2,c2 = divmod(goal_idx, N)
        s += abs(r1-r2)+abs(c1-c2)
    return s

# --------------------------- Problem 2: Grid Maze ---------------------------
# State representation: (r,c)

def grid_neighbors(r, c, grid):
    R = len(grid)
    C = len(grid[0])
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#':
            yield (nr, nc)


def grid_neighbors_cost(state, grid):
    r,c = state
    for nb in grid_neighbors(r,c,grid):
        yield (nb, 1)


def manhattan_grid(state, goal):
    (r,c) = state
    (gr,gc) = goal
    return abs(r-gr) + abs(c-gc)

# --------------------------- Problem 3: Missionaries and Cannibals ---------------------------
# State: (M_left, C_left, boat_side) boat_side: 0 left, 1 right
# Implicitly M_right = M_total - M_left


def mc_neighbors(state, total):
    M_left, C_left, boat = state
    M_right = total - M_left
    C_right = total - C_left
    moves = []
    # possible transports when boat on left (boat=0) => send to right
    # when boat on right (boat=1) => send to left
    options = [(1,0),(2,0),(0,1),(0,2),(1,1)]
    if boat == 0:
        for m,c in options:
            if m <= M_left and c <= C_left:
                nM_left = M_left - m
                nC_left = C_left - c
                # check safety
                if (nM_left == 0 or nM_left >= nC_left) and ((total-nM_left) == 0 or (total-nM_left) >= (total-nC_left)):
                    moves.append((nM_left, nC_left, 1))
    else:
        for m,c in options:
            if m <= M_right and c <= C_right:
                nM_left = M_left + m
                nC_left = C_left + c
                if (nM_left == 0 or nM_left >= nC_left) and ((total-nM_left) == 0 or (total-nM_left) >= (total-nC_left)):
                    moves.append((nM_left, nC_left, 0))
    return moves


def mc_neighbors_cost(state, total):
    for nb in mc_neighbors(state, total):
        yield (nb, 1)


def mc_is_goal(state, total):
    return state[0] == 0 and state[1] == 0 and state[2] == 1

# --------------------------- Instances ---------------------------

def get_taquin_instances():
    # returns list of tuples (start, goal, N, name)
    instances = []
    # 3x3
    start3 = (1,4,2,0,3,6,7,5,8)
    goal3 = tuple([1,2,3,4,5,6,7,8,0])
    instances.append((start3, goal3, 3, 'taquin_3x3'))
    # # 4x4
    # start4 = (5,1,2,3,9,6,0,4,13,10,7,8,14,11,12,15)
    # goal4 = tuple(list(range(1,16))+[0])
    # instances.append((start4, goal4, 4, 'taquin_4x4'))
    # # 5x5 (mild scramble)
    # start5 = (1,2,3,4,5,6,7,8,0,10,11,12,13,9,15,16,17,18,14,20,21,22,23,24,19)
    # goal5 = tuple(list(range(1,25))+[0])
    # instances.append((start5, goal5, 5, 'taquin_5x5'))
    return instances


def get_grid_instances():
    # simple hardcoded small grids; '#' are obstacles, '.' free
    grids = []
    g10 = [
    list("S..#......"),  # 10 chars
    list("##.#.##.#."),  # 10 chars
    list(".........."),  # 10 chars
    list(".####.##.."),  # 10 chars
    list(".........."),  # 10 chars
    list(".#.#.#.#.."),  # 10 chars
    list("....#....."),  # 10 chars
    list("#.######.."),  # 10 chars
    list(".........."),  # 10 chars
    list(".##.####..")   # 10 chars
   ]
    grids.append((g10, (0,0), (4,9), 'grid_10x10'))
    # Larger random-like grids can be built programmatically
    def gen_grid(R,C, density=0.2, seed=1):
        import random
        random.seed(seed)
        g = [[ '.' for _ in range(C)] for __ in range(R)]
        for i in range(R):
            for j in range(C):
                if random.random() < density:
                    g[i][j] = '#'
        g[0][0] = 'S'
        g[R-1][C-1] = 'G'
        return g
    grids.append((gen_grid(20,20,0.22,2), (0,0), (19,19), 'grid_20x20'))
    grids.append((gen_grid(50,50,0.22,3), (0,0), (49,49), 'grid_50x50'))
    return grids


def get_mc_instances():
    return [ (3, 'mc_3'), (4, 'mc_4'), (5, 'mc_5') ]

# --------------------------- Runner / Benchmarker ---------------------------

def run_taquin_all(csv_writer):
    instances = get_taquin_instances()
    for start, goal, N, name in instances:
        print('\nRunning Taquin', name)
        # BFS
        res = measure(lambda metrics=None: bfs(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors(s,N), metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'BFS', N, res)
        # DFS (with depth limit to avoid blow up)
        res = measure(lambda metrics=None: dfs(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors(s,N), limit=50, metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'DFS', N, res)
        # ID
        res = measure(lambda metrics=None: iterative_deepening(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors(s,N), max_depth=80, metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'ID', N, res)
        # UCS
        res = measure(lambda metrics=None: ucs(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors_cost(s,N), metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'UCS', N, res)
        # A* (Manhattan)
        res = measure(lambda metrics=None: a_star(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors_cost(s,N), lambda s: heuristic_manhattan(s,goal,N), metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'A*', N, res)
        # IDA*
        res = measure(lambda metrics=None: ida_star(start, lambda s: sliding_goal(s, goal), lambda s: sliding_neighbors_cost(s,N), lambda s: heuristic_manhattan(s,goal,N), metrics=metrics))
        write_csv_row(csv_writer, 'taquin', name, 'IDA*', N, res)

def run_grid_all(csv_writer):
    instances = get_grid_instances()
    for grid, S, G, name in instances:
        print('\nRunning Grid', name)
        start = S
        goal = G
        # convert grid char matrix
        # BFS
        res = measure(lambda metrics=None: bfs(start, lambda s: s==goal, lambda s: list(grid_neighbors(s[0], s[1], grid)), metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'BFS', len(grid), res)
        # DFS (limited depth)
        res = measure(lambda metrics=None: dfs(start, lambda s: s==goal, lambda s: list(grid_neighbors(s[0],s[1],grid)), limit=1000, metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'DFS', len(grid), res)
        # ID
        res = measure(lambda metrics=None: iterative_deepening(start, lambda s: s==goal, lambda s: list(grid_neighbors(s[0],s[1],grid)), max_depth=2000, metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'ID', len(grid), res)
        # UCS
        res = measure(lambda metrics=None: ucs(start, lambda s: s==goal, lambda s: grid_neighbors_cost(s, grid), metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'UCS', len(grid), res)
        # A* (Manhattan)
        res = measure(lambda metrics=None: a_star(start, lambda s: s==goal, lambda s: grid_neighbors_cost(s, grid), lambda s: manhattan_grid(s, goal), metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'A*', len(grid), res)
        # IDA*
        res = measure(lambda metrics=None: ida_star(start, lambda s: s==goal, lambda s: grid_neighbors_cost(s, grid), lambda s: manhattan_grid(s, goal), metrics=metrics))
        write_csv_row(csv_writer, 'grid', name, 'IDA*', len(grid), res)


def run_mc_all(csv_writer):
    instances = get_mc_instances()
    for total, name in instances:
        print('\nRunning Missionaries-Cannibals', name)
        start = (total, total, 0)
        # BFS
        res = measure(lambda metrics=None: bfs(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors(s,total), metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'BFS', total, res)
        # DFS
        res = measure(lambda metrics=None: dfs(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors(s,total), limit=200, metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'DFS', total, res)
        # ID
        res = measure(lambda metrics=None: iterative_deepening(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors(s,total), max_depth=200, metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'ID', total, res)
        # UCS
        res = measure(lambda metrics=None: ucs(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors_cost(s,total), metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'UCS', total, res)
        # A*
        # naive heuristic: remaining people on left (admissible)
        res = measure(lambda metrics=None: a_star(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors_cost(s,total), lambda s: s[0]+s[1], metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'A*', total, res)
        # IDA*
        res = measure(lambda metrics=None: ida_star(start, lambda s: mc_is_goal(s,total), lambda s: mc_neighbors_cost(s,total), lambda s: s[0]+s[1], metrics=metrics))
        write_csv_row(csv_writer, 'mc', name, 'IDA*', total, res)

# --------------------------- CSV writer ---------------------------

CSV_HEADER = [
    'problem_family', 'instance_name', 'algorithm', 'size_param',
    'sol_length', 'time_s', 'mem_kb_peak', 'mem_kb_current', 'nodes_expanded', 'nodes_generated'
]


def write_csv_row(writer, family, instance, algo, size, measure_dict):
    res = measure_dict['result']
    sol_len = len(res) - 1 if res is not None else -1
    row = [family, instance, algo, size, sol_len, round(measure_dict['time_s'],6), round(measure_dict['mem_kb_peak'],2), round(measure_dict['mem_kb_current'],2), measure_dict['nodes_expanded'], measure_dict['nodes_generated']]
    writer.writerow(row)
    print('\t', algo, 'time=', round(measure_dict['time_s'],3), 's', 'mem_peak_kb=', round(measure_dict['mem_kb_peak'],2), 'nodes_expanded=', measure_dict['nodes_expanded'])

# --------------------------- Main ---------------------------

def main(output_csv='benchmark_results.csv'):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        run_taquin_all(writer)
        run_grid_all(writer)
        run_mc_all(writer)
    print('\nBenchmark finished. Results saved to', output_csv)

if __name__ == '__main__':
    main()
