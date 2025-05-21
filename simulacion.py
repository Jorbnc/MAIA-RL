from typing import Tuple, List, Callable
from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np
import time


def crear_scheduler_oscilante(epsilon_inicial, max_episodios, n_ciclos) -> Callable[[float, int], float]:
    """
    Retorna una función de oscilación para epsilon: epsilon = scheduler(_, episodio)
      - Oscila durante n_ciclos a lo largo de los max_episodios
      - La amplitud decae de forma lineal hasta 0
    """
    def scheduler(prev_eps: float, episodio: int) -> float:
        envelope = 1 - episodio / max_episodios
        cos_term = 0.5 * (1 + np.cos(2 * np.pi * n_ciclos * episodio / max_episodios))
        return epsilon_inicial * envelope * cos_term
    return scheduler


def run(
    tablero, agente, episodios,
    # Ciclos de oscilación para epsilon
    epsilon_ciclos=5,
    # Parámetros del reporte de Qvalores, política y animación del entrenamiento
    print_Qvalores_politica=False, animacion=False, interval=100
) -> None:
    """
    Correr simulación del tablero y entrenamiento del agente.
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
    epsilon_episodico = [agente.epsilon]

    osc_scheduler = crear_scheduler_oscilante(epsilon_inicial=agente.epsilon, max_episodios=episodios, n_ciclos=epsilon_ciclos)

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

        # EPSILON DECAY ------------------------------------------------
        # Proporcional al avance de la simulación (epsilon=0 al final)
        # agente.epsilon *= (episodios - episodio) / episodios

        # Cada cierto número fijo de episodios
        # if episodio % 50 == 0:
        #     agente.epsilon *= 0.9

        # Cosine Annealing
        agente.epsilon = osc_scheduler(agente.epsilon, episodio)
        epsilon_episodico.append(agente.epsilon)
        # -------------------------------------------------------------

    print(f"Completado en {time.time() - time_s:.4f} segundos")

    if print_Qvalores_politica:
        Qvals, politica = agente.obtener_Qmax_politica()
        print("\nQ-tabla:")
        print("Se asigna numpy.nan (en lugar de 0) a celdas terminales y celdas no exploradas\n", Qvals)
        print("\nPolítica Óptima:\n' ' = celda terminal o no explorada")
        print("'X' = movimiento único en escalera/rodadero\n", politica)

    if animacion:
        plot_tablero(tablero, agente_params, Q_values_lista, trayectoria_estado, epsilon_episodico, interval)

    return reward_historico, agente.Qtabla
