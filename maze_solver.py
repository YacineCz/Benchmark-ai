# Labyrinthe simple résolu par BFS / A* / IDA*
import random
from search_algs import measure_run, bfs, astar, ida_star
import math

# Grid: 0 = free, 1 = wall
def generate_grid(n, p_wall=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    grid = [[0 if random.random()>p_wall else 1 for _ in range(n)] for _ in range(n)]
    grid[0][0] = 0
    grid[n-1][n-1] = 0
    return grid

def print_grid(grid, path_coords=None):
    n = len(grid)
    s = ""
    for i in range(n):
        for j in range(n):
            if path_coords and (i,j) in path_coords:
                s += " * "
            elif grid[i][j]==1:
                s += " # "
            elif (i,j)==(0,0):
                s += " S "
            elif (i,j)==(n-1,n-1):
                s += " G "
            else:
                s += " . "
        s += "\n"
    print(s)

def neighbors(pos, grid):
    n = len(grid)
    i,j = pos
    for di,dj,act in [(-1,0,'U'),(1,0,'D'),(0,-1,'L'),(0,1,'R')]:
        ni, nj = i+di, j+dj
        if 0<=ni<n and 0<=nj<n and grid[ni][nj]==0:
            yield ( (ni,nj), act, 1 )

def succ_for_bfs(state):
    pos = state
    return [ (child, act) for (child,act,c) in neighbors(pos, GLOBAL_GRID) ]

def succ_for_astar(state):
    pos = state
    return [ (child, act, c) for (child,act,c) in neighbors(pos, GLOBAL_GRID) ]

def h_manhattan_pos(pos):
    n = len(GLOBAL_GRID)
    gi, gj = n-1, n-1
    return abs(pos[0]-gi)+abs(pos[1]-gj)

def h_euclid_pos(pos):
    n = len(GLOBAL_GRID)
    gi, gj = n-1, n-1
    return math.hypot(pos[0]-gi, pos[1]-gj)

def goal_test(pos):
    n = len(GLOBAL_GRID)
    return pos == (n-1, n-1)

def run_example(n=15, p=0.25, seed=1):
    global GLOBAL_GRID
    GLOBAL_GRID = generate_grid(n, p, seed)
    print("Grid:")
    print_grid(GLOBAL_GRID)
    start = (0,0)
    print("BFS:")
    res = measure_run(bfs, start, goal_test, succ_for_bfs)
    print_summary(res)
    print("A* (Manhattan):")
    res = measure_run(astar, start, goal_test, succ_for_astar, h_manhattan_pos)
    print_summary(res)
    print("IDA* (Manhattan):")
    res = measure_run(ida_star, start, goal_test, succ_for_astar, h_manhattan_pos)
    print_summary(res)

def print_summary(res):
    if res.path is None:
        print("Pas de solution trouvée.")
    else:
        # reconstruct coords from actions (for visualization)
        print(f"Actions len: {len(res.path)}")
    print(f"Temps: {res.time_s:.4f}s, Mémoire peak: {res.mem_kb:.1f} KB")
    print(f"n_expanded: {res.nodes_expanded}, n_generated: {res.nodes_generated}\n")

if __name__ == "__main__":
    run_example(15, 0.25, seed=42)
