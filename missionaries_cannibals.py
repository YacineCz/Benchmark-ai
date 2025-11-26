# État: (M_left, C_left, boat_pos) boat_pos: 0 = left, 1 = right
from search_algs import measure_run, bfs, astar, ida_star
import math

def valid_state(state):
    M_left, C_left, boat = state
    M_right = 3 - M_left
    C_right = 3 - C_left
    # counts must be 0..3
    if not (0<=M_left<=3 and 0<=C_left<=3):
        return False
    # left side: if M_left>0 then M_left >= C_left or C_left==0
    if M_left>0 and C_left> M_left:
        return False
    if M_right>0 and C_right> M_right:
        return False
    return True

def successors(state):
    M, C, boat = state
    res = []
    moves = [(1,0),(2,0),(0,1),(0,2),(1,1)]
    for dm, dc in moves:
        if boat==0: # move from left to right
            nm = M - dm
            nc = C - dc
            nboat = 1
        else:
            nm = M + dm
            nc = C + dc
            nboat = 0
        new = (nm, nc, nboat)
        if valid_state(new):
            res.append((new, f"{dm}M_{dc}C_{'L->R' if boat==0 else 'R->L'}", 1))
    return res

def succ_for_bfs(state):
    return [(child, action) for child,action,c in successors(state)]

def goal_test(state):
    return state == (0,0,1)  # all on right, boat right

def heuristic_simple(state):
    # number of people left divided by boat capacity (2) ceil
    M, C, b = state
    people_left = M + C
    return math.ceil(people_left / 2.0)

def run_example():
    start = (3,3,0)
    print("BFS:")
    r = measure_run(bfs, start, goal_test, succ_for_bfs)
    print_stats(r)
    print("A* (simple heuristic):")
    r = measure_run(astar, start, goal_test, successors, heuristic_simple)
    print_stats(r)
    print("IDA*:")
    r = measure_run(ida_star, start, goal_test, successors, heuristic_simple)
    print_stats(r)

def print_stats(r):
    if r.path is None:
        print("Pas de solution.")
    else:
        print(f"Actions ({len(r.path)}): {r.path}")
    print(f"Temps: {r.time_s:.4f}s, Mémoire peak: {r.mem_kb:.1f} KB")
    print(f"n_expanded: {r.nodes_expanded}, n_generated: {r.nodes_generated}\n")

if __name__ == "__main__":
    run_example()
