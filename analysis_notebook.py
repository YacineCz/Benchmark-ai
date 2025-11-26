import pandas as pd
import matplotlib.pyplot as plt

CSV = 'results.csv'

def main():
    df = pd.read_csv(CSV)
    display_df = df.copy()
    print('Aperçu des résultats:')
    print(display_df.head(30))

    # Nettoyage et conversions
    df_ok = df[df.status=='OK'].copy()
    df_ok['time_s'] = df_ok['time_s'].astype(float)
    df_ok['mem_kb'] = df_ok['mem_kb'].astype(float)
    df_ok['solution_len'] = pd.to_numeric(df_ok['solution_len'], errors='coerce')

    # Temps moyen par algorithme
    grouped = df_ok.groupby('algorithm').agg({'time_s':['mean','median','std'], 'nodes_expanded:generated':'count'})
    print('\nTemps moyen par algorithme:')
    print(grouped)

    # Graphe: temps vs algorithm (boxplot)
    plt.figure()
    df_ok.boxplot(column='time_s', by='algorithm')
    plt.title('Distribution du temps par algorithme')
    plt.suptitle('')
    plt.ylabel('time (s)')
    plt.show()

    # Graphe: solution_len vs algorithm
    plt.figure()
    df_ok.boxplot(column='solution_len', by='algorithm')
    plt.title('Longueur des solutions par algorithme')
    plt.suptitle('')
    plt.ylabel('solution length (moves)')
    plt.show()

if __name__=='__main__':
    main()