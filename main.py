from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run

##
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celda_victoria=100,
    celdas_perdida=[16, 50, 80, 96],
    celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
    celdas_rodadero=[(44, 18), (73, 60), (92, 86)],
)
agente = AgenteQLearning(tab, alpha=0.75, epsilon=0.75, gamma=0.99)
run(tab, agente, episodios=80, animacion=True, plot_reward_acumulado=False)
