from collections import deque
import random
import math

#--- Helpers généraux ---

def index2rc(idx, size):
    return divmod(idx, size)

def rc2index(r,c,size):
    return r*size + c

#--- Génération d'états aléatoires solvables ---

def random_state(size, seed=None):
    # crée une permutation aléatoire mais solvable
    if seed is not None:
        random.seed(seed)
    arr = list(range(size*size))
    random.shuffle(arr)
    if not is_solvable(arr, size):
        # swap two tiles (non-zero) pour changer la parité
        if size*size>=2:
            if arr[0]==0 or arr[1]==0:
                arr[-1], arr[-2] = arr[-2], arr[-1]
            else:
                arr[0], arr[1] = arr[1], arr[0]
    return tuple(arr)

def is_solvable(arr, size):
    # retourne True si la permutation est solvable pour puzzle size x size
    inv = 0
    flat = [x for x in arr if x!=0]
    for i in range(len(flat)):
        for j in range(i+1,len(flat)):
            if flat[i]>flat[j]: inv += 1
    if size % 2 == 1:
        return inv % 2 == 0
    else:
        # blank row from bottom (1-indexed)
        blank_idx = arr.index(0)
        r, c = index2rc(blank_idx, size)
        blank_row_from_bottom = size - r
        return (inv + blank_row_from_bottom) % 2 == 0

#--- Successors génériques (avec coût) ---

def successors_n(state, size):
    res = []
    zero = state.index(0)
    r, c = index2rc(zero, size)
    moves = []
    if r>0: moves.append((-1,0,'Up'))
    if r<size-1: moves.append((1,0,'Down'))
    if c>0: moves.append((0,-1,'Left'))
    if c<size-1: moves.append((0,1,'Right'))
    for dr, dc, action in moves:
        nr, nc = r+dr, c+dc
        nz = rc2index(nr,nc,size)
        lst = list(state)
        lst[zero], lst[nz] = lst[nz], lst[zero]
        res.append((tuple(lst), action, 1))
    return res

def successors_no_cost_n(state, size):
    return [(child, action) for (child,action,c) in successors_n(state,size)]

#--- heuristiques ---

def h_misplaced_n(state, size):
    goal = tuple(list(range(1,size*size))+[0])
    return sum(1 for i,v in enumerate(state) if v!=0 and v!=goal[i])

def h_manhattan_n(state, size):
    dist = 0
    for i,v in enumerate(state):
        if v==0: continue
        goal_idx = v-1
        r1,c1 = index2rc(i,size)
        r2,c2 = index2rc(goal_idx,size)
        dist += abs(r1-r2)+abs(c1-c2)
    return dist

#--- API-compatible pour 3x3 (noms originaux) ---
GOAL_3 = tuple([1,2,3,4,5,6,7,8,0])

def goal_test(state):
    return state == GOAL_3

def successors(state):
    return successors_n(state, 3)

def successors_no_cost(state):
    return successors_no_cost_n(state, 3)

def h_manhattan(state):
    return h_manhattan_n(state, 3)

#--- API pour n x n ---

def goal_test_n(state, size):
    goal = tuple(list(range(1,size*size))+[0])
    return state == goal

def successors_n_wrapper(state, size):
    return successors_n(state, size)


# Alias pour import clair
successors_no_cost_n = successors_no_cost_n
h_manhattan_n = h_manhattan_n