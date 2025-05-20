# Librerías importadas
import matplotlib.pyplot as plt
import numpy as np
# Módulos del proyecto
from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
from helpers import leer_tablero

## Tablero para el proyecto ===============================================
escaleras, rodaderos, perdida, victoria = leer_tablero('tablero.txt')
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=victoria,
    celdas_perdida=perdida,
    celdas_escalera=escaleras,
    celdas_rodadero=rodaderos,
    r_victoria=100, r_perdida=-100, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.65, epsilon=0.25, gamma=1)
run(tab, agente, episodios=45, animacion=True, print_Qtabla_politica=True)

# Imprimir Q-tabla detallada
print("\nQ-tabla detallada (1 significa moverse a la celda mayor inmediata, -1 significa lo contrario):")
for (sa, q) in sorted(agente.Q.items()):
    print(sa, f"{q:.2f}")

## Tablero adicional 1 ====================================================
# tab = Tablero(
#     nro_filas=10,
#     nro_columnas=10,
#     celdas_victoria=[100],
#     celdas_perdida=[16, 50, 80, 96],
#     celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
#     celdas_rodadero=[(38, 18), (73, 60), (92, 86)],
#     r_victoria=100, r_perdida=-100, r_otros=-1
# )
# agente = AgenteQLearning(tab, alpha=0.75, epsilon=0.05, gamma=1)
# run(tab, agente, episodios=75, print_Qtabla_politica=True, animacion=True)
