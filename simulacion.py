from typing import Tuple, List, Callable
from visualizacion import plot_tablero
import matplotlib.pyplot as plt
import numpy as np
import time


def crear_scheduler_oscilante(epsilon_inicial, max_episodios, n_ciclos) -> Callable[[float, int], float]:
    """
    Retorna una función de oscilación para epsilon: epsilon = scheduler(episodio)
      - Oscila durante n_ciclos a lo largo de los max_episodios
      - La amplitud decae de forma lineal hasta 0
    """
    def scheduler(episodio: int) -> float:
        # Controlar la amplitud
        envelope = 1 - episodio / max_episodios
        # Oscilación
        cos_term = 0.5 * (1 + np.cos(2 * np.pi * n_ciclos * episodio / max_episodios))
        # Valor epsilon basado en los dos componentes anteriores
        return epsilon_inicial * envelope * cos_term
    return scheduler


def run(
    tablero, agente, episodios,
    # Ciclos de oscilación para epsilon
    epsilon_ciclos=5,
    # Parámetros del reporte de Qvalores, política y animación del entrenamiento
    print_Qvalores_politica=False, animacion=False, interval=100,
    # Otros logs
    plot_pasos=False
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
    pasos_historico = []

    osc_scheduler = crear_scheduler_oscilante(
        epsilon_inicial=agente.epsilon, max_episodios=episodios, n_ciclos=epsilon_ciclos
    )

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

        # Registrar logs
        Q_values_lista.append(agente.max_Q_values())
        reward_historico.append(sum(reward_acumulado))
        pasos_historico.append(pasos)

        # EPSILON DECAY ------------------------------------------------
        # Proporcional al avance de la simulación (epsilon=0 al final)
        # agente.epsilon *= (episodios - episodio) / episodios
        # epsilon_episodico.append(agente.epsilon)

        # Cada cierto número fijo de episodios
        # if episodio % 100 == 0:
        #     agente.epsilon *= 0.9
        # epsilon_episodico.append(agente.epsilon)

        # Cosine Annealing
        agente.epsilon = osc_scheduler(episodio)
        epsilon_episodico.append(agente.epsilon)
        # -------------------------------------------------------------

    print(f"Completado en {time.time() - time_s:.4f} segundos")

    if plot_pasos:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 4))
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(1, episodios + 1)
            for spine in ax.spines.values():
                spine.set_alpha(0.25)
        ax1.semilogy(range(1, episodios + 1), pasos_historico, alpha=1, color='black', label="Número de pasos")
        ax2.plot(range(1, episodios + 1), epsilon_episodico[:-1], color='black', alpha=1, label="Reward acumulado")
        ax3.plot(range(1, episodios + 1), reward_historico, color='blue', alpha=1, label="Reward acumulado")
        ax3.set_yscale('symlog')
        fsize = 13
        ax1.set_ylabel("Pasos", fontsize=fsize)
        ax2.set_ylabel("Epsilon", fontsize=fsize)
        ax3.set_ylabel("Reward", fontsize=fsize)
        ax1.set_xticks([])
        ax2.set_xticks([])
        plt.tight_layout()
        plt.show()

    if print_Qvalores_politica:
        Qvals, politica = agente.obtener_Qmax_politica()
        print("\nQ-valores máximos:")
        print("Se asigna numpy.nan (en lugar de 0) a celdas terminales y celdas no exploradas\n", Qvals)
        print("\nPolítica:\n' ' = celda terminal o no explorada")
        print("'X' = movimiento único en escalera/rodadero\n", politica)

    if animacion:
        plot_tablero(tablero, agente_params, Q_values_lista, trayectoria_estado, epsilon_episodico, interval)

    return reward_historico, agente.Qtabla
