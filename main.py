from tablero import Tablero
from visualizacion import plot_tablero
from simulacion import run
import time

##
tab = Tablero(
    nro_filas=10,
    nro_columnas=10,
    celda_victoria=100,
    celdas_perdida=[16, 50, 80, 96],
    celdas_escalera=[(14, 46), (21, 77), (25, 36), (68, 90), (84, 100)],
    celdas_rodadero=[(44, 18), (73, 60), (93, 87)],
)

start = time.time()
run(tab, episodios=100)
end = time.time()
print(end - start)
