"""
Visualization script for benchmark_results.csv
Generates clear comparative charts using matplotlib.
- Bar charts for time, memory, nodes expanded.
- Grouped by problem and algorithm.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Load CSV ----------------
def load_results(csv_path='benchmark_results.csv'):
    return pd.read_csv(csv_path)

# ---------------- Plotting helpers ----------------
def plot_bar(df, title, ylabel, value_col, save_name):
    plt.figure(figsize=(12,6))
    # Grouped by algorithm average
    grouped = df.groupby('algorithm')[value_col].mean().sort_values()

    grouped.plot(kind='bar')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_by_problem(df, problem_family, value_col, ylabel, save_name):
    sub = df[df['problem_family'] == problem_family]
    plt.figure(figsize=(12,6))

    algos = sub['algorithm'].unique()
    instances = sub['instance_name'].unique()
    x = np.arange(len(instances))
    width = 0.12

    for i, algo in enumerate(algos):
        vals = []
        for inst in instances:
            part = sub[(sub['instance_name'] == inst) & (sub['algorithm'] == algo)]
            vals.append(part[value_col].values[0] if not part.empty else 0)
        plt.bar(x + i*width, vals, width, label=algo)

    plt.title(f"{problem_family} – {ylabel}")
    plt.ylabel(ylabel)
    plt.xticks(x + width*len(algos)/2, instances, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# ---------------- Master function ----------------
def generate_all_plots(csv_path='benchmark_results.csv'):
    df = load_results(csv_path)

    # Global averages
    plot_bar(df, 'Temps moyen par algorithme', 'Temps (sec)', 'time_s', 'plot_time_avg.png')
    plot_bar(df, 'Mémoire peak moyenne par algorithme', 'Mémoire (KB)', 'mem_kb_peak', 'plot_memory_avg.png')
    plot_bar(df, 'Noeuds expandés par algorithme', 'Nodes expanded', 'nodes_expanded', 'plot_nodes_avg.png')

    # Per problem family
    families = df['problem_family'].unique()
    for fam in families:
        plot_by_problem(df, fam, 'time_s', 'Temps (sec)', f'{fam}_time.png')
        plot_by_problem(df, fam, 'mem_kb_peak', 'Mémoire peak (KB)', f'{fam}_memory.png')
        plot_by_problem(df, fam, 'nodes_expanded', 'Nodes expanded', f'{fam}_nodes.png')

    print("All plots saved (PNG files).")

# ---------------- Main ----------------
if __name__ == '__main__':
    generate_all_plots()