# Librerías importadas
import matplotlib.pyplot as plt
import numpy as np
# Módulos del proyecto
from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
from helpers import leer_tablero

## Tablero para el proyecto ===============================================
rodaderos, escaleras, perdida, victoria = leer_tablero('tablero.txt')
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=victoria,
    celdas_perdida=perdida,
    celdas_escalera=escaleras,
    celdas_rodadero=rodaderos,
    r_victoria=100, r_perdida=-100, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.5, epsilon=1, gamma=1)
run(tab, agente, episodios=60, epsilon_ciclos=4, animacion=True)

# Una vez entrenado el agente, podemos acceder a su Q-tabla (valor máximo) y política
Qtabla, politica = agente.obtener_Qtabla_politica()

# También se puede imprimir Q-tabla detallada
# print("\nQ-tabla detallada (1 significa moverse a la celda mayor inmediata, -1 significa lo contrario):")
# for (sa, q) in sorted(agente.Q.items()):
#     print(sa, f"{q:.2f}")

## Tablero adicional 1 ====================================================
tab = Tablero(
    nro_columnas=10,
    nro_filas=10,
    celdas_victoria=[96],
    celdas_perdida=[14, 56, 85],
    celdas_escalera=[(6, 26), (7, 70), (22, 58), (60, 80), (68, 93), (84, 100), (47, 65), (33, 51)],
    celdas_rodadero=[(25, 20), (30, 13), (57, 36), (73, 61)],
    r_victoria=100, r_perdida=-200, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.5, epsilon=1, gamma=1)
run(tab, agente, episodios=400, epsilon_ciclos=4, animacion=True, interval=50)
