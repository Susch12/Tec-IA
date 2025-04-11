# visualizador.py

import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from correccionEuristica import AlgoritmoGeneraciones

def get_generations_data(num_generations):
    algoritmo = AlgoritmoGeneraciones()
    return algoritmo.run(num_generations)

def main():
    root = tk.Tk()
    root.withdraw()

    num_generaciones = simpledialog.askinteger(
        "Número de Generaciones",
        "Ingresa el número de generaciones:"
    )

    if num_generaciones is None or num_generaciones <= 0:
        print("Valor inválido. Saliendo del programa.")
        return

    generaciones_x, generaciones_y, generaciones_z = get_generations_data(num_generaciones)

    z_ultima = generaciones_z[-1]
    indice_mejor = np.argmax(z_ultima)

    mejor_x = generaciones_x[-1][indice_mejor]
    mejor_y = generaciones_y[-1][indice_mejor]
    mejor_z = generaciones_z[-1][indice_mejor]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for gen in range(num_generaciones):
        ax.scatter(
            generaciones_x[gen],
            generaciones_y[gen],
            generaciones_z[gen],
            label=f'Gen {gen}'
        )

    ax.scatter(
        [mejor_x], [mejor_y], [mejor_z],
        s=100, marker='*', edgecolor='black',
        label='Mejor solución'
    )

    ax.set_xlabel('Activo A (X)')
    ax.set_ylabel('Activo B (Y)')
    ax.set_zlabel('Fitness (Z)')
    plt.title('Evolución de Portafolios - Algoritmo Genético')
    ax.legend()

    print(f"\nMejor portafolio encontrado:")
    print(f"  Activo A: {mejor_x:.2%}")
    print(f"  Activo B: {mejor_y:.2%}")
    print(f"  Fitness: {mejor_z:.4f}")
    plt.show()

if __name__ == '__main__':
    main()

