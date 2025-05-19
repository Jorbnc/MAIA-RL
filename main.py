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
)
agente = AgenteQLearning(tab, alpha=0.65, epsilon=0.25, gamma=1)
run(tab, agente, 100, animacion=True, print_politica=True)

### Tablero adicional 1
# tab = Tablero(
#    nro_filas=10,
#    nro_columnas=10,
#    celdas_victoria=[100],
#    celdas_perdida=[16, 50, 80, 96],
#    celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
#    celdas_rodadero=[(38, 18), (73, 60), (92, 86)],
# )
# agente = AgenteQLearning(tab, alpha=0.65, epsilon=0.05, gamma=1)
# run(tab, agente, 75, animacion=True, print_politica=True)
