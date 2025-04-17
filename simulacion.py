from agente import AgenteQLearning
from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np


def run(tablero, episodios=1000) -> None:
    """Correr simulación y plotear la última trayectoria."""

    agente = AgenteQLearning(tablero, epsilon=0.1)
    trayectoria_ult = []
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

            # WARNING: La evaluación durante cada iteración es innecesaria.
            # Probar solo con cambiar a: for episodio in range(episodios - 1)...

            # Evaluar si hay condición de finalización
            if estado_siguiente == agente.tablero.celda_victoria or estado_siguiente in agente.tablero.celdas_perdida:

                # Para plotear la última trayectoria
                if episodio == episodios - 1:
                    trayectoria_ult = trayectoria

                print(f"Episodio {episodio + 1} terminó en {pasos} pasos, con reward {reward}.")
                break

    # plt.plot(np.cumsum(reward_acumulado))
    # plot_tablero(tablero, trayectoria_ult)
