import time
import tracemalloc
import csv
import heapq
import math
from collections import deque, defaultdict
import itertools
import json
import os

# --------------------------- Outils de mesure ---------------------------

class Metrics:
    
    # Classe pour suivre les statistiques de recherche :
    # nodes_expanded : nombre de nœuds dont on a exploré les successeurs
    # nodes_generated : nombre de nœuds créés / ajoutés dans la frontière
   
    def __init__(self):
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def reset(self):
        
        # Réinitialise les compteurs à zéro.
        # Utile si l'on veut réutiliser le même objet Metrics pour plusieurs exécutions
        
        self.nodes_expanded = 0
        self.nodes_generated = 0


def measure(func, *args, **kwargs):
    
    # Mesure le temps, la mémoire et les métriques d'un algorithme de recherche.

    # Arguments :
    # func : fonction à exécuter (algorithme de recherche)
    # *args, **kwargs : arguments optionnels à passer à la fonction

    # Retourne un dictionnaire contenant :
    # result : le chemin solution retourné par l'algorithme
    # time_s : temps d'exécution en secondes
    # mem_kb_peak : mémoire maximale utilisée (KB)
    # mem_kb_current : mémoire courante à la fin (KB)
    # nodes_expanded : nombre de nœuds étendus
    # nodes_generated : nombre de nœuds générés
    
    tracemalloc.start()  # démarre le suivi mémoire
    t0 = time.perf_counter()  # démarre le chronomètre
    metrics = Metrics()  # initialise un objet Metrics pour collecter les stats
    result = func(*args, metrics=metrics, **kwargs)  # exécute la fonction en passant metrics
    t1 = time.perf_counter()  # temps de fin
    current, peak = tracemalloc.get_traced_memory()  # récupère la mémoire actuelle et maximale
    tracemalloc.stop()  # arrête le suivi mémoire

    # retourne toutes les mesures sous forme de dictionnaire
    return {
        'result': result,
        'time_s': t1 - t0,
        'mem_kb_peak': peak / 1024.0,       # conversion en kilo-octets
        'mem_kb_current': current / 1024.0, # conversion en kilo-octets
        'nodes_expanded': metrics.nodes_expanded,
        'nodes_generated': metrics.nodes_generated,
    }



# ---------------------------  Algorithmes de recherche ---------------------------

def bfs(start, is_goal, get_neighbors, metrics=None):
    # Si aucun objet metrics n'est fourni, on en crée un.
    # metrics permet de compter le nombre de nœuds générés, explorés, etc.
    if metrics is None: metrics = Metrics()

    # La structure de données principale est une file (FIFO),
    # car l'algorithme BFS explore d'abord les nœuds de plus faible profondeur.
    frontier = deque([start])

    # Le dictionnaire came_from mémorise le parent de chaque nœud.
    # Il sert à reconstruire le chemin une fois la solution trouvée.
    came_from = {start: None}

    # On incrémente le nombre de nœuds générés : l'état initial.
    metrics.nodes_generated += 1

    # Tant que la file n'est pas vide, il reste encore des nœuds à explorer.
    while frontier:
        # On récupère (et enlève) le premier nœud de la file.
        # C'est un traitement FIFO → BFS.
        node = frontier.popleft()

        # On incrémente le compteur de nœuds développés,
        # c'est-à-dire le nombre d'états réellement explorés.
        metrics.nodes_expanded += 1

        # Si le nœud courant est l'objectif, on reconstruit le chemin
        # en remontant grâce à came_from et on retourne ce chemin.
        if is_goal(node):
            return reconstruct_path(came_from, node)

        # Sinon, on génère ses voisins (successeurs possibles)
        for neighbor in get_neighbors(node):
            # Chaque voisin est considéré comme un nouveau nœud généré.
            metrics.nodes_generated += 1

            # Si le voisin n'a jamais été exploré auparavant,
            # on l'enregistre et on l'ajoute dans la file d'attente.
            # Cela évite les boucles et redondances.
            if neighbor not in came_from:
                came_from[neighbor] = node
                frontier.append(neighbor)

    # Si la file est vide et aucun objectif n’a été trouvé,
    # la recherche échoue → on retourne None.
    return None

def dfs(start, is_goal, get_neighbors, limit=None, metrics=None):
    # Si aucun objet "metrics" n'est fourni, on en crée un.
    # Cet objet permet de compter le nombre de nœuds générés / explorés.
    if metrics is None: metrics = Metrics()

    # Ensemble des nœuds déjà visités.
    # Il permet d’éviter les cycles infinis et le retraitement de mêmes états.
    visited = set()

    # La structure utilisée est une pile (stack) car DFS explore en profondeur.
    # Chaque élément contient : (nœud, parent, profondeur)
    stack = [(start, None, 0)]

    # Le dictionnaire "parents" permet de reconstruire le chemin
    # une fois qu’on atteint l’état but.
    parents = {start: None}

    # On compte le nœud initial comme généré.
    metrics.nodes_generated += 1

    # Tant que la pile n’est pas vide, il reste des chemins à explorer.
    while stack:
        # On dépile le dernier élément ajouté (LIFO) → recherche en profondeur.
        node, parent, depth = stack.pop()

        # Si le nœud a déjà été exploré, on l’ignore.
        if node in visited:
            continue

        # On marque le nœud comme visité pour éviter les répétitions.
        visited.add(node)

        # On incrémente le compteur de nœuds développés (explorés réellement).
        metrics.nodes_expanded += 1

        # Si l’état courant correspond à l’objectif,
        # on reconstruit et retourne le chemin depuis le nœud de départ.
        if is_goal(node):
            return reconstruct_path(parents, node)

        # Si une limite de profondeur est définie (DFS limité),
        # on arrête le développement de ce nœud lorsqu’elle est atteinte.
        if limit is not None and depth >= limit:
            continue

        # Sinon, on développe les successeurs du nœud courant.
        for neighbor in get_neighbors(node):
            # Chaque successeur est considéré comme un nœud généré.
            metrics.nodes_generated += 1

            # On n’empile que les nœuds non visités afin d’éviter les boucles.
            if neighbor not in visited:
                # On sauvegarde le parent pour permettre la reconstruction du chemin.
                parents[neighbor] = node

                # On ajoute le voisin à la pile, avec un niveau de profondeur +1.
                stack.append((neighbor, node, depth + 1))

    # Si aucun chemin n’atteint l’objectif,
    # l’algorithme échoue et on retourne None.
    return None


def iterative_deepening(start, is_goal, get_neighbors, max_depth=50, metrics=None):
    # Si aucun objet "metrics" n'est fourni, on en crée un.
    # Cet objet servira à compter le nombre total de nœuds générés et explorés.
    if metrics is None: metrics = Metrics()

    # Boucle sur les profondeurs de 0 jusqu'à max_depth.
    # Chaque itération correspond à un DFS limité à une profondeur spécifique.
    for depth in range(max_depth + 1):
        # On crée un objet Metrics séparé pour cette itération
        # afin de compter les nœuds générés et explorés à cette profondeur.
        iter_metrics = Metrics()

        # On appelle DFS limité à la profondeur "depth".
        res = dfs(start, is_goal, get_neighbors, limit=depth, metrics=iter_metrics)

        # On accumule les statistiques de cette itération dans les métriques globales.
        metrics.nodes_expanded += iter_metrics.nodes_expanded
        metrics.nodes_generated += iter_metrics.nodes_generated

        # Si DFS a trouvé une solution à cette profondeur, on la retourne.
        if res is not None:
            return res

    # Si aucune solution n’a été trouvée jusqu'à max_depth, on retourne None.
    return None


def ucs(start, is_goal, get_neighbors_cost, metrics=None):
    # Si aucun objet "metrics" n'est fourni, on en crée un.
    # Cet objet permet de compter le nombre de nœuds générés et explorés.
    if metrics is None: metrics = Metrics()

    # La structure principale est une file de priorité (heap),
    # qui permet de toujours extraire le nœud avec le coût total le plus faible.
    # Chaque élément est un tuple : (coût_total, nœud)
    frontier = []
    heapq.heappush(frontier, (0, start))

    # Dictionnaire qui stocke le coût minimal trouvé jusqu'à chaque nœud.
    cost_so_far = {start: 0}

    # Dictionnaire des parents pour reconstruire le chemin.
    came_from = {start: None}

    # On compte le nœud initial comme généré.
    metrics.nodes_generated += 1

    # Tant que la file de priorité n'est pas vide, on explore les nœuds.
    while frontier:
        # On dépile le nœud avec le coût total le plus faible (UCS).
        cost, node = heapq.heappop(frontier)

        # On incrémente le compteur de nœuds développés.
        metrics.nodes_expanded += 1

        # Si le nœud courant est l'objectif, on reconstruit le chemin.
        if is_goal(node):
            return reconstruct_path(came_from, node)

        # On examine tous les voisins du nœud courant.
        # get_neighbors_cost(node) renvoie des tuples : (voisin, coût du pas)
        for neighbor, step_cost in get_neighbors_cost(node):
            # Chaque voisin est compté comme généré.
            metrics.nodes_generated += 1

            # Calcul du coût total pour atteindre ce voisin.
            new_cost = cost_so_far[node] + step_cost

            # Si le voisin n’a jamais été visité ou si un meilleur coût est trouvé :
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                # On met à jour le coût minimal pour atteindre ce voisin.
                cost_so_far[neighbor] = new_cost

                # On ajoute le voisin dans la file de priorité avec le nouveau coût.
                heapq.heappush(frontier, (new_cost, neighbor))

                # On enregistre le parent pour reconstruire le chemin.
                came_from[neighbor] = node

    # Si aucun chemin n’atteint l’objectif, on retourne None.
    return None

def a_star(start, is_goal, get_neighbors_cost, heuristic, metrics=None):
    # Si aucun objet "metrics" n'est fourni, on en crée un.
    # Il servira à compter le nombre de nœuds générés et explorés.
    if metrics is None: metrics = Metrics()

    # La structure principale est une file de priorité (heap),
    # triée selon f = g + h (coût total estimé pour atteindre le but).
    # Chaque élément est un tuple : (f_score, g_score, nœud)
    frontier = []

    # Calcul de l'heuristique pour l'état initial.
    start_h = heuristic(start)
    heapq.heappush(frontier, (start_h, 0, start))

    # Dictionnaire des parents pour reconstruire le chemin une fois l'objectif atteint.
    came_from = {start: None}

    # Dictionnaire g_score : coût réel minimal trouvé pour atteindre chaque nœud.
    g_score = {start: 0}

    # On compte le nœud initial comme généré.
    metrics.nodes_generated += 1

    # Tant que la file de priorité n'est pas vide, on explore les nœuds.
    while frontier:
        # On dépile le nœud avec le plus petit f_score (g + h)
        f, g, node = heapq.heappop(frontier)

        # On incrémente le compteur de nœuds développés (explorés réellement)
        metrics.nodes_expanded += 1

        # Si le nœud courant est l'objectif, on reconstruit le chemin.
        if is_goal(node):
            return reconstruct_path(came_from, node)

        # On examine tous les voisins du nœud courant.
        # get_neighbors_cost(node) renvoie des tuples : (voisin, coût du pas)
        for neighbor, step_cost in get_neighbors_cost(node):
            # Chaque voisin est considéré comme un nœud généré.
            metrics.nodes_generated += 1

            # Calcul du coût réel pour atteindre ce voisin via le nœud courant.
            tentative_g = g_score[node] + step_cost

            # Si le voisin n’a jamais été visité ou si un meilleur coût est trouvé :
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # On met à jour le coût réel minimal pour atteindre ce voisin.
                g_score[neighbor] = tentative_g

                # Calcul du f_score : g (coût réel) + h (heuristique)
                f_score = tentative_g + heuristic(neighbor)

                # On ajoute le voisin à la file de priorité avec son f_score
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

                # On enregistre le parent pour reconstruire le chemin final.
                came_from[neighbor] = node

    # Si aucun chemin n’atteint l’objectif, on retourne None.
    return None


def ida_star(start, is_goal, get_neighbors_cost, heuristic, metrics=None, max_iterations=100000):
    # Si aucun objet "metrics" n'est fourni, on en crée un.
    # Cet objet permet de compter le nombre de nœuds générés et explorés.
    if metrics is None: metrics = Metrics()

    # La borne initiale (bound) est fixée à l'heuristique de l'état de départ.
    bound = heuristic(start)

    # Le chemin courant est initialisé avec le nœud de départ.
    path = [start]

    # Dictionnaire g pour mémoriser le coût réel jusqu'à chaque nœud dans le chemin.
    g = {start: 0}

    # Fonction récursive de recherche limitée par la borne f (g+h)
    def search(path, g_cost, bound):
        # On récupère le dernier nœud du chemin courant.
        node = path[-1]

        # Calcul du f = g + h pour le nœud courant.
        f = g_cost + heuristic(node)

        # Si f dépasse la borne actuelle, on retourne f pour ajuster la prochaine itération.
        if f > bound:
            return f, None

        # Si le nœud courant est l'objectif, on retourne le chemin trouvé.
        if is_goal(node):
            return 'FOUND', list(path)

        # Initialisation du seuil minimum pour la prochaine itération.
        min_threshold = float('inf')

        # On explore tous les voisins du nœud courant.
        for neighbor, step_cost in get_neighbors_cost(node):
            # Chaque voisin est compté comme généré.
            metrics.nodes_generated += 1

            # On ignore les voisins déjà présents dans le chemin pour éviter les cycles.
            if neighbor in path:
                continue

            # On ajoute le voisin au chemin.
            path.append(neighbor)

            # On incrémente le compteur de nœuds développés (explorés réellement)
            metrics.nodes_expanded += 1

            # Calcul du coût réel pour atteindre le voisin.
            g_next = g_cost + step_cost

            # Appel récursif sur le voisin avec le coût mis à jour et la même borne.
            t, res = search(path, g_next, bound)

            # Si on a trouvé la solution, on la retourne immédiatement.
            if t == 'FOUND':
                return 'FOUND', res

            # Sinon, on met à jour le seuil minimum pour la prochaine itération.
            if t < min_threshold:
                min_threshold = t

            # On retire le voisin du chemin pour explorer d'autres branches.
            path.pop()

        # Retourne le seuil minimum trouvé dans cette itération.
        return min_threshold, None

    # Compteur d'itérations pour éviter les boucles infinies.
    iterations = 0

    # Boucle principale de l'IDA*, qui ajuste la borne à chaque itération.
    while True:
        iterations += 1
        # Limite de sécurité pour éviter un nombre trop élevé d'itérations.
        if iterations > max_iterations:
            return None

        # On lance la recherche avec la borne actuelle.
        t, res = search(path, 0, bound)

        # Si la solution est trouvée, on la retourne.
        if t == 'FOUND':
            return res

        # Si aucune solution n'est possible (seuil infini), on retourne None.
        if t == float('inf'):
            return None

        # On met à jour la borne pour la prochaine itération.
        bound = t


def reconstruct_path(came_from, node):
    
    # Reconstruit le chemin solution à partir d'un dictionnaire de parents (came_from).
    # Arguments :
    # came_from : dictionnaire {nœud: parent} indiquant d'où chaque nœud a été atteint
    # node : nœud final (objectif atteint)
    # Retour : path : liste ordonnée des nœuds depuis le départ jusqu'au but
    
    path = []
    cur = node  # commence à partir du nœud final
    while cur is not None:
        path.append(cur)          # ajoute le nœud courant au chemin
        cur = came_from.get(cur)  # remonte vers le parent
    path.reverse()  # inverse la liste pour obtenir le chemin du départ vers le but
    return path



# --------------------------- Problème 1 : Taquin  ---------------------------
# L'état est représenté par un tuple d'entiers de longueur N*N
# où 0 représente la case vide.

def sliding_neighbors(state, N):
    # On trouve l'indice de la case vide (0)
    zero_idx = state.index(0)

    # Conversion de l'indice linéaire en coordonnées (ligne, colonne)
    r, c = divmod(zero_idx, N)

    moves = []

    # Déplacements possibles : haut, bas, gauche, droite
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc

        # Vérification que le déplacement reste dans le plateau
        if 0 <= nr < N and 0 <= nc < N:
            # Conversion des coordonnées en indice linéaire
            ni = nr * N + nc

            # Création d'une nouvelle configuration en échangeant 0 et la case cible
            new = list(state)
            new[zero_idx], new[ni] = new[ni], new[zero_idx]

            # On ajoute ce nouvel état à la liste des successeurs
            moves.append(tuple(new))
    
    # Retourne la liste de tous les voisins possibles
    return moves


def sliding_neighbors_cost(state, N):
    # Générateur qui renvoie tous les voisins avec un coût associé
    # Ici, chaque déplacement a un coût de 1
    for nb in sliding_neighbors(state, N):
        yield (nb, 1)


def sliding_goal(state, goal):
    # Vérifie si l'état courant correspond à l'état objectif
    return state == goal


def heuristic_misplaced(state, goal):
    # Heuristique : nombre de cases mal placées (hors de la case vide)
    return sum(1 for a,b in zip(state, goal) if a != b and a != 0)


def heuristic_manhattan(state, goal, N):
    # Heuristique : somme des distances de Manhattan de chaque case
    # jusqu'à sa position dans l'état objectif
    # On crée un dictionnaire pour retrouver rapidement l'indice objectif de chaque valeur
    pos_goal = {val: i for i, val in enumerate(goal)}

    s = 0
    for idx, val in enumerate(state):
        if val == 0:  # On ignore la case vide
            continue

        # Indice cible de la valeur actuelle
        goal_idx = pos_goal[val]

        # Conversion des indices linéaires en coordonnées (ligne, colonne)
        r1, c1 = divmod(idx, N)
        r2, c2 = divmod(goal_idx, N)

        # Distance de Manhattan pour cette case
        s += abs(r1 - r2) + abs(c1 - c2)
    
    return s

# --------------------------- Problème 2 : Labyrinthe en grille ---------------------------
# La représentation d'un état est un tuple (ligne, colonne) : (r, c)

def grid_neighbors(r, c, grid):
    # R : nombre de lignes, C : nombre de colonnes
    R = len(grid)
    C = len(grid[0])

    # Déplacements possibles : haut, bas, gauche, droite
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc

        # Vérification que le déplacement reste dans les limites de la grille
        # et que la case cible n'est pas un mur (représenté par '#')
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != '#':
            # On retourne le voisin valide
            yield (nr, nc)


def grid_neighbors_cost(state, grid):
    # Générateur qui renvoie les voisins de l'état courant avec un coût
    # Ici, chaque déplacement a un coût de 1
    r, c = state
    for nb in grid_neighbors(r, c, grid):
        yield (nb, 1)


def manhattan_grid(state, goal):
    # Heuristique : distance de Manhattan entre l'état courant et l'état objectif
    r, c = state
    gr, gc = goal
    return abs(r - gr) + abs(c - gc)


# --------------------------- Problème 3 : Missionnaires et Cannibales ---------------------------
# Représentation d'un état : (M_left, C_left, boat_side)
# boat_side : 0 = bateau à gauche, 1 = bateau à droite
# Implicitement : M_right = M_total - M_left, C_right = C_total - C_left

def mc_neighbors(state, total):
    # Extraction des valeurs de l'état courant
    M_left, C_left, boat = state
    M_right = total - M_left
    C_right = total - C_left

    moves = []

    # Options possibles de transport : (missionnaires, cannibales)
    options = [(1,0),(2,0),(0,1),(0,2),(1,1)]

    if boat == 0:  # Bateau à gauche => envoyer des personnes à droite
        for m, c in options:
            if m <= M_left and c <= C_left:
                nM_left = M_left - m
                nC_left = C_left - c

                # Vérification de sécurité :
                # Sur chaque rive, les missionnaires ne doivent pas être en minorité
                if (nM_left == 0 or nM_left >= nC_left) and \
                   ((total - nM_left) == 0 or (total - nM_left) >= (total - nC_left)):
                    # Ajouter le nouvel état avec le bateau à droite
                    moves.append((nM_left, nC_left, 1))
    else:  # Bateau à droite => envoyer des personnes à gauche
        for m, c in options:
            if m <= M_right and c <= C_right:
                nM_left = M_left + m
                nC_left = C_left + c

                # Vérification de sécurité
                if (nM_left == 0 or nM_left >= nC_left) and \
                   ((total - nM_left) == 0 or (total - nM_left) >= (total - nC_left)):
                    # Ajouter le nouvel état avec le bateau à gauche
                    moves.append((nM_left, nC_left, 0))

    # Retourne la liste de tous les voisins possibles
    return moves


def mc_neighbors_cost(state, total):
    # Générateur qui renvoie les voisins avec un coût
    # Chaque déplacement a un coût de 1
    for nb in mc_neighbors(state, total):
        yield (nb, 1)


def mc_is_goal(state, total):
    # Vérifie si tous les missionnaires et cannibales sont passés à droite
    # et que le bateau est également à droite
    return state[0] == 0 and state[1] == 0 and state[2] == 1

# --------------------------- Instances des problèmes ---------------------------

def get_taquin_instances():
    # Retourne une liste de tuples pour les instances du Taquin.
    # Chaque tuple contient : (état de départ, état objectif, taille N, nom)

    instances = []

    # 3x3
    start3 = (1,4,2,0,3,6,7,5,8)
    goal3 = tuple([1,2,3,4,5,6,7,8,0])
    instances.append((start3, goal3, 3, 'taquin_3x3'))

    # # 4x4
    # start4 = (5,1,2,3,9,6,0,4,13,10,7,8,14,11,12,15)
    # goal4 = tuple(list(range(1,16)) + [0])
    # instances.append((start4, goal4, 4, 'taquin_4x4'))

    # # 5x5 (scramble modéré)
    # start5 = (1,2,3,4,5,6,7,8,0,10,11,12,13,9,15,16,17,18,14,20,21,22,23,24,19)
    # goal5 = tuple(list(range(1,25)) + [0])
    # instances.append((start5, goal5, 5, 'taquin_5x5'))

    return instances


def get_grid_instances():
    # Retourne une liste de tuples pour les instances de labyrinthe en grille.
    # Chaque tuple contient : (grille, état de départ, état objectif, nom)
    # '#' représente un obstacle, '.' un espace libre.
    grids = []

    # Grille 10x10 codée à la main
    g10 = [
        list("S..#......"),  # ligne 0
        list("##.#.##.#."),  # ligne 1
        list(".........."),  # ligne 2
        list(".####.##.."),  # ligne 3
        list(".........."),  # ligne 4
        list(".#.#.#.#.."),  # ligne 5
        list("....#....."),  # ligne 6
        list("#.######.."),  # ligne 7
        list(".........."),  # ligne 8
        list(".##.####..")   # ligne 9
    ]
    grids.append((g10, (0,0), (4,9), 'grid_10x10'))

    # Générateur de grandes grilles aléatoires
    def gen_grid(R, C, density=0.2, seed=1):
        # Génère une grille R x C avec des obstacles aléatoires.
        # density : probabilité d'avoir un obstacle sur une case
        # seed : pour reproductibilité
        # 'S' = start, 'G' = goal
        import random
        random.seed(seed)
        g = [['.' for _ in range(C)] for __ in range(R)]
        for i in range(R):
            for j in range(C):
                if random.random() < density:
                    g[i][j] = '#'
        g[0][0] = 'S'        # départ
        g[R-1][C-1] = 'G'    # objectif
        return g

    # Grilles plus grandes générées automatiquement
    grids.append((gen_grid(20,20,0.22,2), (0,0), (19,19), 'grid_20x20'))
    grids.append((gen_grid(50,50,0.22,3), (0,0), (49,49), 'grid_50x50'))

    return grids


def get_mc_instances():
    # Retourne les instances du problème Missionnaires & Cannibales.
    # Chaque tuple contient : (nombre de missionnaires/cannibales, nom)
    return [(3, 'mc_3'), (4, 'mc_4'), (5, 'mc_5')]

# --------------------------- Exécuteur / Benchmarker ---------------------------
# Cette fonction exécute tous les algorithmes de recherche pour chaque instance de Taquin
# et écrit les résultats dans un fichier CSV via le writer fourni.

def run_taquin_all(csv_writer):
    # Récupère toutes les instances de Taquin (3x3, 4x4, 5x5, etc.)
    instances = get_taquin_instances()
    
    # Boucle sur chaque instance
    for start, goal, N, name in instances:
        print('\nRunning Taquin', name)  # Affiche le nom de l'instance en cours pour le suivi

        # ---------------- BFS (Breadth-First Search) ----------------
        # Recherche en largeur : explore tous les nœuds d'un niveau avant de passer au suivant
        res = measure(
            lambda metrics=None: bfs(
                start,                                 # État initial
                lambda s: sliding_goal(s, goal),      # Fonction qui teste si l'état est le but
                lambda s: sliding_neighbors(s, N),    # Fonction qui génère les voisins de l'état
                metrics=metrics                        # Mesures de performance (temps, mémoire, nœuds)
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'BFS', N, res)  # Écriture des résultats dans le CSV

        # ---------------- DFS (Depth-First Search) limité ----------------
        # Recherche en profondeur : explore un chemin complet avant de revenir en arrière
        # Limite de profondeur fixée à 50 pour éviter explosion combinatoire
        res = measure(
            lambda metrics=None: dfs(
                start,
                lambda s: sliding_goal(s, goal),
                lambda s: sliding_neighbors(s, N),
                limit=50,                              # Limite de profondeur
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'DFS', N, res)

        # ---------------- ID (Iterative Deepening) ----------------
        # DFS itérative : combine avantages DFS et BFS
        # Augmente progressivement la profondeur maximale jusqu'à max_depth
        res = measure(
            lambda metrics=None: iterative_deepening(
                start,
                lambda s: sliding_goal(s, goal),
                lambda s: sliding_neighbors(s, N),
                max_depth=80,                           # Profondeur maximale
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'ID', N, res)

        # ---------------- UCS (Uniform Cost Search) ----------------
        # Recherche de coût uniforme : choisit toujours le nœud avec le coût total le plus faible
        res = measure(
            lambda metrics=None: ucs(
                start,
                lambda s: sliding_goal(s, goal),
                lambda s: sliding_neighbors_cost(s, N), # Génération des voisins avec coût
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'UCS', N, res)

        # ---------------- A*----------------
        # Recherche A* : utilise une heuristique (ici Manhattan) pour guider la recherche
        res = measure(
            lambda metrics=None: a_star(
                start,
                lambda s: sliding_goal(s, goal),
                lambda s: sliding_neighbors_cost(s, N),
                lambda s: heuristic_manhattan(s, goal, N), # Heuristique pour estimer la distance au but
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'A*', N, res)

        # ---------------- IDA* (Iterative Deepening A*) ----------------
        # A* itératif : combine A* et recherche en profondeur itérative
        # Permet de limiter la mémoire tout en gardant l’efficacité d’A*
        res = measure(
            lambda metrics=None: ida_star(
                start,
                lambda s: sliding_goal(s, goal),
                lambda s: sliding_neighbors_cost(s, N),
                lambda s: heuristic_manhattan(s, goal, N),
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'taquin', name, 'IDA*', N, res)


def run_grid_all(csv_writer):
    # Exécute tous les algorithmes de recherche pour chaque instance de grille
    # et écrit les résultats dans le CSV via le writer fourni.
    # Récupération de toutes les instances de grille
    instances = get_grid_instances()
    
    # Boucle sur chaque instance de grille
    for grid, S, G, name in instances:
        print('\nRunning Grid', name)  # Affiche le nom de l'instance pour suivi
        
        start = S   # État de départ (coordonnées de départ)
        goal = G    # État but (coordonnées de la case objectif)
        
        # ---------------- BFS (Breadth-First Search) ----------------
        # Explore tous les nœuds d'un niveau avant de passer au suivant
        res = measure(
            lambda metrics=None: bfs(
                start,
                lambda s: s == goal,                         # Fonction test but
                lambda s: list(grid_neighbors(s[0], s[1], grid)), # Génère les voisins dans la grille
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'BFS', len(grid), res)

        # ---------------- DFS (Depth-First Search) limité ----------------
        # Explore un chemin complet avant de revenir en arrière
        # Limite de profondeur fixée à 1000 pour éviter explosion combinatoire
        res = measure(
            lambda metrics=None: dfs(
                start,
                lambda s: s == goal,
                lambda s: list(grid_neighbors(s[0], s[1], grid)),
                limit=1000,
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'DFS', len(grid), res)

        # ---------------- ID (Iterative Deepening) ----------------
        # DFS itérative : combine avantages DFS et BFS
        # Augmente progressivement la profondeur maximale jusqu'à max_depth
        res = measure(
            lambda metrics=None: iterative_deepening(
                start,
                lambda s: s == goal,
                lambda s: list(grid_neighbors(s[0], s[1], grid)),
                max_depth=2000,
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'ID', len(grid), res)

        # ---------------- UCS (Uniform Cost Search) ----------------
        # Recherche de coût uniforme : choisit toujours le nœud avec le coût total le plus faible
        res = measure(
            lambda metrics=None: ucs(
                start,
                lambda s: s == goal,
                lambda s: grid_neighbors_cost(s, grid),  # Génère voisins avec coût
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'UCS', len(grid), res)

        # ---------------- A*----------------
        # Recherche A* : utilise une heuristique (ici Manhattan) pour guider la recherche
        res = measure(
            lambda metrics=None: a_star(
                start,
                lambda s: s == goal,
                lambda s: grid_neighbors_cost(s, grid),
                lambda s: manhattan_grid(s, goal),  # Heuristique Manhattan
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'A*', len(grid), res)

        # ---------------- IDA* (Iterative Deepening A*) ----------------
        # A* itératif : combine A* et recherche en profondeur itérative
        # Permet de limiter la mémoire tout en gardant l’efficacité d’A*
        res = measure(
            lambda metrics=None: ida_star(
                start,
                lambda s: s == goal,
                lambda s: grid_neighbors_cost(s, grid),
                lambda s: manhattan_grid(s, goal),
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'grid', name, 'IDA*', len(grid), res)

def run_mc_all(csv_writer):
    # Exécute tous les algorithmes de recherche pour chaque instance du problème Missionnaires & Cannibales
    # et écrit les résultats dans le CSV via le writer fourni.

    # Récupération de toutes les instances M&C (3, 4, 5 missionnaires/cannibales)
    instances = get_mc_instances()
    
    # Boucle sur chaque instance
    for total, name in instances:
        print('\nRunning Missionaries-Cannibals', name)  # Affiche le nom de l'instance pour suivi
        
        start = (total, total, 0)  # État initial : tous les missionnaires et cannibales à gauche, bateau à gauche (0)
        
        # ---------------- BFS (Breadth-First Search) ----------------
        # Explore tous les nœuds d'un niveau avant de passer au suivant
        res = measure(
            lambda metrics=None: bfs(
                start,
                lambda s: mc_is_goal(s, total),   # Fonction qui teste si l'état est le but
                lambda s: mc_neighbors(s, total), # Génère les voisins possibles depuis l'état courant
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'BFS', total, res)

        # ---------------- DFS (Depth-First Search) limité ----------------
        # Explore un chemin complet avant de revenir en arrière
        # Limite de profondeur fixée à 200 pour éviter explosion combinatoire
        res = measure(
            lambda metrics=None: dfs(
                start,
                lambda s: mc_is_goal(s, total),
                lambda s: mc_neighbors(s, total),
                limit=200,
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'DFS', total, res)

        # ---------------- ID (Iterative Deepening) ----------------
        # DFS itérative : augmente progressivement la profondeur maximale jusqu'à max_depth
        res = measure(
            lambda metrics=None: iterative_deepening(
                start,
                lambda s: mc_is_goal(s, total),
                lambda s: mc_neighbors(s, total),
                max_depth=200,
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'ID', total, res)

        # ---------------- UCS (Uniform Cost Search) ----------------
        # Recherche par coût uniforme : choisit toujours le nœud avec le coût total le plus faible
        res = measure(
            lambda metrics=None: ucs(
                start,
                lambda s: mc_is_goal(s, total),
                lambda s: mc_neighbors_cost(s, total),  # Génère voisins avec coût
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'UCS', total, res)

        # ---------------- A*----------------
        # Recherche A* : utilise une heuristique simple
        # Heuristique naïve ici : nombre de personnes restantes sur la rive gauche (admissible)
        res = measure(
            lambda metrics=None: a_star(
                start,
                lambda s: mc_is_goal(s, total),
                lambda s: mc_neighbors_cost(s, total),
                lambda s: s[0] + s[1],  # Heuristique : missionnaires + cannibales restants
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'A*', total, res)

        # ---------------- IDA* (Iterative Deepening A*) ----------------
        # A* itératif : combine A* et recherche en profondeur itérative
        res = measure(
            lambda metrics=None: ida_star(
                start,
                lambda s: mc_is_goal(s, total),
                lambda s: mc_neighbors_cost(s, total),
                lambda s: s[0] + s[1],  # Même heuristique que A*
                metrics=metrics
            )
        )
        write_csv_row(csv_writer, 'mc', name, 'IDA*', total, res)


# --------------------------- CSV---------------------------
# Définition de l'entête du fichier CSV
# Chaque ligne contiendra les informations suivantes :
CSV_HEADER = [
    'problem_family',    # famille du problème (taquin, grid, mc)
    'instance_name',     # nom de l'instance
    'algorithm',         # nom de l'algorithme utilisé
    'size_param',        # paramètre de taille (ex: N pour taquin ou taille de grille)
    'sol_length',        # longueur de la solution trouvée (-1 si pas de solution)
    'time_s',            # temps d'exécution en secondes
    'mem_kb_peak',       # mémoire maximale utilisée (KB)
    'mem_kb_current',    # mémoire actuelle à la fin (KB)
    'nodes_expanded',    # nombre de nœuds développés
    'nodes_generated'    # nombre de nœuds générés
]

def write_csv_row(writer, family, instance, algo, size, measure_dict):
    # Écrit une ligne dans le fichier CSV avec les mesures de performance.
    # measure_dict doit contenir : result, time_s, mem_kb_peak, mem_kb_current, nodes_expanded, nodes_generated

    res = measure_dict['result']                        # solution trouvée
    sol_len = len(res) - 1 if res is not None else -1   # longueur de la solution (-1 si aucune solution)
    
    # Création de la ligne à écrire dans le CSV
    row = [
        family,
        instance,
        algo,
        size,
        sol_len,
        round(measure_dict['time_s'], 6),          # temps arrondi à 6 décimales
        round(measure_dict['mem_kb_peak'], 2),    # mémoire maximale arrondie
        round(measure_dict['mem_kb_current'], 2), # mémoire actuelle arrondie
        measure_dict['nodes_expanded'],
        measure_dict['nodes_generated']
    ]
    
    writer.writerow(row)  # écriture de la ligne dans le CSV

    # Affichage console pour suivi
    print(
        '\t', algo,
        'time=', round(measure_dict['time_s'], 3), 's',
        'mem_peak_kb=', round(measure_dict['mem_kb_peak'], 2),
        'nodes_expanded=', measure_dict['nodes_expanded']
    )


# --------------------------- Main ---------------------------
def main(output_csv='benchmark_results.csv'):
    # Point d'entrée principal.
    # Crée le fichier CSV, exécute tous les benchmarks et sauvegarde les résultats.

    # Ouverture du fichier CSV en mode écriture
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)  # écriture de l'entête

        # Exécution des benchmarks pour chaque famille de problème
        run_taquin_all(writer)
        run_grid_all(writer)
        run_mc_all(writer)

    print('\nBenchmark terminé. Résultats enregistrés dans', output_csv)


# Exécution du script si appelé directement
if __name__ == '__main__':
    main()
