from typing import Tuple, List
from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np
import time


def run(tablero, agente, episodios, print_politica=False, animacion=False) -> None:
    """
    Correr simulación.
    """

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
    agente_params = (agente.alpha, agente.epsilon, agente.gamma)
    Q_values_lista = []
    reward_historico = []

    # Episodios
    time_s = time.time()
    for episodio in range(1, episodios + 1):

        # Reiniciar condiciones al inicio de cada episodio
        agente.pos = 1
        trayectoria_estado = [agente.pos]
        reward_acumulado = []
        pasos = 0

        # Recorrer tablero
        while True:
            estado, accion, reward, estado_siguiente = step()
            pasos += 1
            trayectoria_estado.append(estado_siguiente)
            reward_acumulado.append(reward)

            # Evaluar si hay condición de finalización
            if estado_siguiente in tablero.celdas_victoria:
                print(f"Episodio {episodio} (✅) terminó en {pasos} pasos, con reward {sum(reward_acumulado)}.")
                break
            elif estado_siguiente in tablero.celdas_perdida:
                print(f"Episodio {episodio} (❌) terminó en {pasos} pasos, con reward {sum(reward_acumulado)}.")
                break

        # Registrar Q-values
        Q_values_lista.append(agente.max_Q_values())
        reward_historico.append(sum(reward_acumulado))

        # Epsilon decay
        agente.epsilon *= (episodios - episodio) / episodios

    print(f"Completado en {time.time() - time_s:.4f} segundos")

    if print_politica:
        agente.print_Q_politica()

    if animacion:
        plot_tablero(tablero, agente_params, Q_values_lista, trayectoria_estado)

    return reward_historico
