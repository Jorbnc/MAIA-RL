from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
import matplotlib.pyplot as plt
import numpy as np
from tablero import coord_a_celda

##
# tab = Tablero(
#     nro_filas=10,
#     nro_columnas=10,
#     celdas_victoria=[100],
#     celdas_perdida=[16, 50, 80, 96],
#     celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
#     celdas_rodadero=[(44, 18), (73, 60), (92, 86)],
# )
#

##


def cambio_coord(coord, nro_cols=10, nro_filas=10):
    fila, col = coord
    x = col + 1
    y = nro_filas - fila
    return x, y


##
with open('tablero.txt', 'r') as f:
    primera_linea = f.readline().strip()
    n, m, s, t = map(int, primera_linea.split())
    nro_cols = 10

    # Rodaderos
    rodaderos = []
    for _ in range(n):
        fila1, col1, fila2, col2 = map(int, f.readline().split())
        x_fin, y_fin = cambio_coord((fila1, col1))
        x_inicio, y_inicio = cambio_coord((fila2, col2))
        rodaderos.append((coord_a_celda(x_fin, y_fin), coord_a_celda(x_inicio, y_inicio)))

    # Escaleras
    escaleras = []
    for _ in range(m):
        fila1, col1, fila2, col2 = map(int, f.readline().split())
        x_inicio, y_inicio = cambio_coord((fila1, col1))
        x_fin, y_fin = cambio_coord((fila2, col2))
        escaleras.append((coord_a_celda(x_inicio, y_inicio), coord_a_celda(x_fin, y_fin)))

    # Pérdida
    perdida = []
    for _ in range(s):
        fila1, col1 = map(int, f.readline().split())
        x, y = cambio_coord((fila1, col1))
        perdida.append(coord_a_celda(x, y))

    # Victoria
    victoria = []
    for _ in range(t):
        fila1, col1 = map(int, f.readline().split())
        x, y = cambio_coord((fila1, col1))
        victoria.append(coord_a_celda(x, y))

##
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=victoria,
    celdas_perdida=perdida,
    celdas_escalera=escaleras,
    celdas_rodadero=rodaderos,
)
agente = AgenteQLearning(tab, alpha=0.75, epsilon=0.25, gamma=1)
run(tab, agente, 60, animacion=True)
