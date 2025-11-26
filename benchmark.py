import csv
import random
from search_algs import measure_run, bfs, dfs, iterative_deepening, astar, ida_star
import time
import os

# Import des modules-problèmes
import eight_puzzle
import maze_solver
import missionaries_cannibals

OUTPUT = 'results.csv'

# Configuration des expériences
EXPERIMENTS = []

# 1) Taquin : 3x3 et 4x4
# for size, seeds in [(3, [1,2,3]), (4, [1,2,3])]:
for size, seeds in [(3, [1,2,3])]:
    for s in seeds:
        EXPERIMENTS.append(('taquin', size, s))

# 2) Labyrinthe : différentes tailles & densités
for n in [10, 15]:
    for p in [0.15, 0.25]:
        for seed in [10, 20]:
            EXPERIMENTS.append(('maze', (n,p), seed))

# 3) Missionnaires-cannibales : problème standard (runs multiples pour stabilité)
for seed in [0,1,2]:
    EXPERIMENTS.append(('mc', None, seed))

# Liste des algorithmes à exécuter (nom, callable, type)
# Type 'uninformed' expects successors without cost (for bfs/dfs/id)
ALGORITHMS = [
    ('BFS', lambda start, goal, succ: measure_run(bfs, start, goal, succ), 'uninformed'),
    ('DFS', lambda start, goal, succ: measure_run(dfs, start, goal, succ), 'uninformed'),
    ('ID', lambda start, goal, succ: measure_run(iterative_deepening, start, goal, succ), 'uninformed'),
    ('A*', None, 'informed'),
    ('IDA*', None, 'informed')
]

# Helpers pour exécuter chaque problème

def run_taquin(size, seed, csv_writer):
    random.seed(seed)
    start = eight_puzzle.random_state(size, seed)
    if size==3:
        print(f"Taquin 3x3 seed={seed}")
    else:
        print(f"Taquin 4x4 seed={seed}")

    # wrappers
    if size==3:
        start_state = start
        goal_test = eight_puzzle.goal_test
        succ_cost = eight_puzzle.successors
        succ_no_cost = eight_puzzle.successors_no_cost
        h = eight_puzzle.h_manhattan
    else:
        start_state = start
        goal_test = lambda s: eight_puzzle.goal_test_n(s, size)
        succ_cost = lambda s: eight_puzzle.successors_n(s, size)
        succ_no_cost = lambda s: eight_puzzle.successors_no_cost_n(s, size)
        h = lambda s: eight_puzzle.h_manhattan_n(s, size)

    for name, wrapper, typ in ALGORITHMS:
        try:
            if typ=='uninformed':
                res = wrapper(start_state, goal_test, succ_no_cost)
            else:
                if name=='A*':
                    res = measure_run(astar, start_state, goal_test, succ_cost, h)
                else:
                    res = measure_run(ida_star, start_state, goal_test, succ_cost, h)
        except Exception as e:
            print(f"Erreur {name} taquin size {size} seed {seed}: {e}")
            res = None

        write_result(csv_writer, 'taquin', size, seed, name, res)


def run_maze(cfg, seed, csv_writer):
    import maze_solver  # <-- Import local propre
    n, p = cfg
    print(f"Maze n={n} p={p} seed={seed}")

    grid = maze_solver.generate_grid(n, p, seed)
    maze_solver.GLOBAL_GRID = grid  # <-- Pas besoin de global ici

    start = (0, 0)
    goal_test = maze_solver.goal_test
    succ_no_cost = maze_solver.succ_for_bfs
    succ_cost = maze_solver.succ_for_astar
    heuristic = maze_solver.h_manhattan_pos

    for name, wrapper, typ in ALGORITHMS:
        try:
            if typ == 'uninformed':
                res = wrapper(start, goal_test, succ_no_cost)
            else:
                if name == 'A*':
                    res = measure_run(astar, start, goal_test, succ_cost, heuristic)
                else:
                    res = measure_run(ida_star, start, goal_test, succ_cost, heuristic)
        except Exception as e:
            print(f"Erreur {name} maze: {e}")
            res = None

        write_result(csv_writer, 'maze', f"{n}x{n}_p{p}", seed, name, res)


def run_mc(_, seed, csv_writer):
    print(f"Missionnaires-Cannibales run seed={seed}")
    start = (3,3,0)
    goal_test = missionaries_cannibals.goal_test
    succ_no_cost = missionaries_cannibals.succ_for_bfs
    succ_cost = missionaries_cannibals.successors
    h = missionaries_cannibals.heuristic_simple

    for name, wrapper, typ in ALGORITHMS:
        try:
            if typ=='uninformed':
                res = wrapper(start, goal_test, succ_no_cost)
            else:
                if name=='A*':
                    res = measure_run(astar, start, goal_test, succ_cost, h)
                else:
                    res = measure_run(ida_star, start, goal_test, succ_cost, h)
        except Exception as e:
            print(f"Erreur {name} mc: {e}")
            res = None
        write_result(csv_writer, 'mc', '3x3', seed, name, res)


def write_result(csv_writer, problem, instance, seed, alg_name, res):
    if res is None or res.path is None:
        row = [problem, instance, seed, alg_name, 'FAIL', '', '', '', '', '']
    else:
        row = [
            problem,
            instance,
            seed,
            alg_name,
            'OK',
            len(res.path),
            res.cost,
            f"{res.time_s:.6f}",
            f"{res.mem_kb:.2f}",
            f"{res.nodes_expanded}:{res.nodes_generated}"
        ]
    csv_writer.writerow(row)
    print(' ->', problem, instance, alg_name, row[4])


if __name__=='__main__':
    # En-tête CSV
    header = ['problem','instance','seed','algorithm','status','solution_len','cost','time_s','mem_kb','nodes_expanded:generated']
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
    with open(OUTPUT,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for kind, arg, seed in EXPERIMENTS:
            if kind=='taquin':
                run_taquin(arg, seed, writer)
            elif kind=='maze':
                run_maze(arg, seed, writer)
            elif kind=='mc':
                run_mc(arg, seed, writer)
    print('Terminé. Résultats ->', OUTPUT)