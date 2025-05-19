# Librerías importadas
import matplotlib.pyplot as plt
import numpy as np
# Módulos del proyecto
from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
from helpers import leer_tablero

## Tablero dado en el curso (semana 7)
escaleras, rodaderos, perdida, victoria = leer_tablero('tablero.txt')
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=victoria,
    celdas_perdida=perdida,
    celdas_escalera=escaleras,
    celdas_rodadero=rodaderos,
    r_victoria=500, r_perdida=-500, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.65, epsilon=0.25, gamma=1)
# run(tab, agente, 100, animacion=True, print_Qtabla_politica=True)

## Tablero adicional 1
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=[100],
    celdas_perdida=[16, 50, 80, 96],
    celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
    celdas_rodadero=[(38, 18), (73, 60), (92, 86)],
    r_victoria=100, r_perdida=-100, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.75, epsilon=0.05, gamma=1)
run(tab, agente, 75, print_Qtabla_politica=True, animacion=True)

## Tablero Tutoría Semana 6
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=[80, 100],
    celdas_perdida=[23, 37, 45, 67, 89],
    celdas_escalera=[(8, 25), (21, 82), (43, 77), (50, 91), (62, 96), (66, 87)],
    celdas_rodadero=[
        (98, 28), (95, 24), (92, 51), (83, 19), (73, 1), (64, 36), (69, 33),
        (59, 17), (55, 7), (52, 11), (44, 22), (46, 5), (48, 9),
    ],
)
agente = AgenteQLearning(tab, alpha=0.65, epsilon=0.05, gamma=1)
# run(tab, agente, 75, animacion=True, print_Qtabla_politica=True)
