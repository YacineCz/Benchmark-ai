import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Chargement du CSV ----------------
def load_results(csv_path='benchmark_results.csv'):
    # Charge les résultats de benchmark depuis un fichier CSV.

    return pd.read_csv(csv_path)


# ---------------- Fonctions utilitaires pour les graphiques ----------------
def plot_bar(df, title, ylabel, value_col, save_name):
    # Trace un diagramme en barres pour comparer les algorithmes selon une métrique donnée.
    # df : DataFrame contenant les résultats
    # title : titre du graphique
    # ylabel : label de l'axe Y
    # value_col : colonne du DataFrame à représenter (ex : 'time_s', 'nodes_expanded')
    # save_name : nom du fichier pour sauvegarder le graphique

    plt.figure(figsize=(12,6))
    # Calcul de la moyenne par algorithme
    grouped = df.groupby('algorithm')[value_col].mean().sort_values()

    grouped.plot(kind='bar')  # trace un bar plot
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')  # rotation des labels pour lisibilité
    plt.tight_layout()
    plt.savefig(save_name)  # sauvegarde du graphique
    plt.close()


def plot_by_problem(df, problem_family, value_col, ylabel, save_name):
    # Trace un diagramme en barres groupées par instance pour un type de problème donné.
    # df : DataFrame contenant les résultats
    # problem_family : nom de la famille de problèmes ('taquin', 'grid', 'mc', etc.)
    # value_col : colonne du DataFrame à représenter
    # ylabel : label de l'axe Y
    # save_name : nom du fichier pour sauvegarder le graphique

    sub = df[df['problem_family'] == problem_family]  # filtre les données pour la famille

    plt.figure(figsize=(12,6))
    algos = sub['algorithm'].unique()         # liste des algorithmes
    instances = sub['instance_name'].unique() # liste des instances
    x = np.arange(len(instances))             # positions des instances sur l'axe X
    width = 0.12                              # largeur de chaque barre

    # Pour chaque algorithme, récupérer les valeurs par instance
    for i, algo in enumerate(algos):
        vals = []
        for inst in instances:
            part = sub[(sub['instance_name'] == inst) & (sub['algorithm'] == algo)]
            vals.append(part[value_col].values[0] if not part.empty else 0)
        plt.bar(x + i*width, vals, width, label=algo)  # ajoute les barres groupées

    plt.title(f"{problem_family} – {ylabel}")
    plt.ylabel(ylabel)
    plt.xticks(x + width*len(algos)/2, instances, rotation=45, ha='right')  # labels centrés
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)  # sauvegarde du graphique
    plt.close()

# ---------------- Fonction principale ----------------
def generate_all_plots(csv_path='benchmark_results.csv'):
    # Génère tous les graphiques à partir des résultats de benchmark CSV.
    # Chargement des résultats depuis le CSV
    df = load_results(csv_path)

    # ---------------- Graphiques globaux par algorithme ----------------
    # Temps moyen par algorithme
    plot_bar(df, 'Temps moyen par algorithme', 'Temps (sec)', 'time_s', 'plot_time_avg.png')
    # Mémoire maximale moyenne par algorithme
    plot_bar(df, 'Mémoire peak moyenne par algorithme', 'Mémoire (KB)', 'mem_kb_peak', 'plot_memory_avg.png')
    # Nombre moyen de nœuds explorés par algorithme
    plot_bar(df, 'Noeuds expandés par algorithme', 'Nodes expanded', 'nodes_expanded', 'plot_nodes_avg.png')

    # ---------------- Graphiques par famille de problèmes ----------------
    families = df['problem_family'].unique()  # liste des familles (taquin, grid, mc, ...)
    for fam in families:
        # Temps par instance
        plot_by_problem(df, fam, 'time_s', 'Temps (sec)', f'{fam}_time.png')
        # Mémoire peak par instance
        plot_by_problem(df, fam, 'mem_kb_peak', 'Mémoire peak (KB)', f'{fam}_memory.png')
        # Nœuds expandés par instance
        plot_by_problem(df, fam, 'nodes_expanded', 'Nodes expanded', f'{fam}_nodes.png')

    print("Tous les graphiques ont été sauvegardés (fichiers PNG).")


# ---------------- Main ----------------
if __name__ == '__main__':
    # Appel de la fonction principale pour générer tous les graphiques
    generate_all_plots()
