import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------
# Charger les données
# ----------------------
file_path = "results.csv"  # change selon ton fichier
df = pd.read_csv(file_path, sep=",")  # tabulation comme séparateur

# ----------------------
# Nettoyer certaines colonnes
# ----------------------
# Séparer nodes_expanded et nodes_generated
df[['nodes_expanded', 'nodes_generated']] = df['nodes_expanded:generated'].str.split(':', expand=True)
df['nodes_expanded'] = pd.to_numeric(df['nodes_expanded'], errors='coerce')
df['nodes_generated'] = pd.to_numeric(df['nodes_generated'], errors='coerce')

# Convertir les autres colonnes numériques
df['solution_len'] = pd.to_numeric(df['solution_len'], errors='coerce')
df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
df['mem_kb'] = pd.to_numeric(df['mem_kb'], errors='coerce')
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

# ----------------------
# Filtrer les échecs
# ----------------------
df_ok = df[df['status'] == 'OK']

# ----------------------
# Créer les graphiques
# ----------------------
metrics = ['solution_len', 'time_s', 'mem_kb', 'nodes_expanded']
titles = {
    'solution_len': "Longueur de la solution",
    'time_s': "Temps d'exécution (s)",
    'mem_kb': "Mémoire utilisée (KB)",
    'nodes_expanded': "Nœuds développés"
}

# Créer un dossier pour sauvegarder les graphiques
os.makedirs("plots", exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(12,6))
    sns.barplot(data=df_ok, x='algorithm', y=metric, hue='problem')
    plt.title(f"{titles[metric]} par algorithme et par problème")
    plt.xlabel("Algorithme")
    plt.ylabel(titles[metric])
    plt.xticks(rotation=30)
    plt.tight_layout()
    # Sauvegarder le graphique
    plt.savefig(f"plots/{metric}_comparison.png")
    plt.show()

# ----------------------
# Tableau comparatif résumé
# ----------------------
summary = df_ok.groupby(['problem', 'algorithm']).agg({
    'solution_len':'mean',
    'time_s':'mean',
    'mem_kb':'mean',
    'nodes_expanded':'mean'
}).reset_index()

print("Résumé comparatif (moyennes par problème et algorithme) :")
print(summary)

# Optionnel : sauvegarder le résumé
summary.to_csv("plots/summary_comparison.csv", index=False)
