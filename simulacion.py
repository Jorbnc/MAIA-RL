from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import time


def run(tablero, agente, episodios) -> None:
    """Correr simulación y plotear la última trayectoria."""

    def step() -> Tuple[int, int, float, int]:
        """
        Representa un paso en la simulación con base en las condiciones actuales.
        Actualiza la Q-table con base en (Sₜ,Aₜ,Rₜ,Sₜ₊₁) y retorna la tupla.
        """

        estado = agente.pos
        accion = agente.escoger_accion(estado)

        # Transición y Reward del paso actual
        estado_siguiente = tablero.transicion(estado, accion)
        reward = tablero.reward(estado_siguiente)

        # Actualizar Q y estado (pos)
        agente.actualizar_Q(estado, accion, reward, estado_siguiente)
        agente.pos = estado_siguiente

        return estado, accion, reward, estado_siguiente

    # Logging
    reward_acumulado = []
    Q_values_lista = []

    # Episodios
    time_s = time.time()
    for episodio in range(1, episodios + 1):

        # Reiniciar condiciones al inicio de cada episodio
        agente.pos = 1
        trayectoria_state = [agente.pos]
        pasos = 0

        # Recorrer tablero
        while True:
            estado, accion, reward, estado_siguiente = step()
            pasos += 1
            trayectoria_state.append(estado_siguiente)
            reward_acumulado.append(reward)

            # Evaluar si hay condición de finalización
            if estado_siguiente == tablero.celda_victoria or estado_siguiente in tablero.celdas_perdida:
                print(f"Episodio {episodio} terminó en {pasos} pasos, con reward {reward}.")
                break

        Q_values_lista.append(agente.Q_values())

        # Epsilon decay
        agente.epsilon *= (episodios - episodio) / episodios

    print(f"Completado en {time.time() - time_s:.4f} segundos")
    # plt.plot(np.cumsum(reward_acumulado))
    plot_tablero(tablero, Q_values_lista=Q_values_lista, trayectoria=trayectoria_state)
    # plot_tablero(tablero, trayectoria=trayectoria_state)
