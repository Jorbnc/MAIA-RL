from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np


def run(tablero, agente, episodios=1000) -> None:
    """Correr simulación y plotear la última trayectoria."""

    reward_acumulado = []

    for episodio in range(episodios):
        # Reiniciar condiciones al inicio de cada episodio
        agente.pos = 1
        trayectoria = [agente.pos]
        pasos = 0

        # Recorrer tablero
        while True:
            estado, accion, reward, estado_siguiente = agente.step()
            pasos += 1
            trayectoria.append(estado_siguiente)
            reward_acumulado.append(reward)

            # Evaluar si hay condición de finalización
            if estado_siguiente == agente.tablero.celda_victoria or estado_siguiente in agente.tablero.celdas_perdida:
                print(f"Episodio {episodio + 1} terminó en {pasos} pasos, con reward {reward}.")
                break

    plt.plot(np.cumsum(reward_acumulado))
    plt.show()
    # plot_tablero(tablero, trayectoria)
