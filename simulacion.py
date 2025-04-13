from agente import Agente_QLearning
from visualizacion import plot_tablero


def run(tablero, episodios=1000) -> None:
    """
    Run a simple simulation where the agent plays the game over multiple episodes.
    Each episode resets the agent's position to 1 and runs until it reaches a terminal state.
    Terminal states are defined as either reaching the victory cell or any lost cell.
    """
    agente = Agente_QLearning(tablero)
    trayectoria_ult = []

    for episodio in range(episodios):
        # Reset agent position at the beginning of each episode
        agente.pos = 1
        trayectoria = [agente.pos]
        steps = 0
        while True:
            estado, accion, reward, estado_siguiente = agente.step()
            steps += 1
            trayectoria.append(estado_siguiente)

            if estado_siguiente == agente.tablero.celda_victoria or estado_siguiente in agente.tablero.celdas_perdida:
                if episodio == episodios - 1:
                    trayectoria_ult = trayectoria
                print(f"Episodio {episodio + 1} terminó en {steps} pasos, con reward {reward}.")
                break

    print(max(agente.Q, key=agente.Q.get))
    plot_tablero(tablero, trayectoria_ult)
