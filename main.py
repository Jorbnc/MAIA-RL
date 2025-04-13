from tablero import Tablero
from visualizacion import plot_tablero
from simulacion import run

##
# tab = Tablero(
#     nro_filas=10,
#     nro_columnas=10,
#     celda_victoria=100,
#     celdas_perdida=[55, 77, 87, 100],
#     celdas_escalera=[(19, 38), (35, 76), (29, 51)],
#     celdas_rodadero=[(27, 5), (99, 58), (89, 68)],
# )
#
# run(tab, episodios=100)

##
tab2 = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celda_victoria=100,
    celdas_perdida=[16, 50, 80, 96],
    celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 97)],
    celdas_rodadero=[(44, 18), (73, 60), (93, 87)],
)

run(tab2, episodios=100)
