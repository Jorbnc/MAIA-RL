# Librerías importadas
import matplotlib.pyplot as plt
import numpy as np
# Módulos del proyecto
from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
from helpers import leer_tablero, guardar_Qtabla, cargar_Qtabla

## Tablero adicional  ====================================================
tab = Tablero(
    nro_columnas=10,
    nro_filas=10,
    celdas_victoria=[96],
    celdas_perdida=[14, 56, 85],
    celdas_escalera=[(6, 26), (7, 70), (22, 58), (60, 80), (68, 93), (84, 100), (47, 65), (33, 51)],
    celdas_rodadero=[(25, 20), (30, 13), (57, 36), (73, 61)],
    r_victoria=100, r_perdida=-100, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.35, epsilon=0.75, gamma=1)

# Run
reward_historico, Qtabla = run(
    tab, agente, episodios=150,
    epsilon_ciclos=5,
    print_Qvalores_politica=True, animacion=True
)

## Corer por más episodios ===============================================
reward_historico, Qtabla = run(
    tab, agente, episodios=1000,
    epsilon_ciclos=5,
    print_Qvalores_politica=True, animacion=False
)
