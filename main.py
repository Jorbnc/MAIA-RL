from tablero import Tablero
from visualizacion import plot_tablero

##
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celda_victoria=100,
    celdas_perdida=[55, 62, 87, 100],
    celdas_escalera=[(19, 38), (35, 76), (29, 51)],
    celdas_rodadero=[(27, 5), (99, 58), (89, 68)],
)

plot_tablero(tab)
