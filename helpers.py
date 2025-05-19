from typing import Tuple, List
from tablero import coord_a_celda


def cambio_coord(coord, nro_cols=10, nro_filas=10):
    """
    Adapta las coordenadas de estilo Numpy (o lista de listas en Python) a
    coordenadas cartesianas que empiezan en 1.
    """
    fila, col = coord
    return col + 1, nro_filas - fila


def leer_tablero(file) -> Tuple[List, List, List, List]:
    """
    Lee la definición del tablero desde un archivo, adapta las coordenadas y las transforma
    a números de celda para ser procesadas de forma adecuada por la clase 'Tablero'
    """
    print(f"Generando tablero desde {file}...")

    with open(file, 'r') as f:
        primera_linea = f.readline().strip()
        # Según especificación del proyecto tenemos:
        # n escaleras, m rodaderos, s pérdidas, y t victorias
        n, m, s, t = map(int, primera_linea.split())

        # Rodaderos
        rodaderos = []
        for _ in range(n):
            fila1, col1, fila2, col2 = map(int, f.readline().split())
            x1, y1 = cambio_coord((fila1, col1))
            x2, y2 = cambio_coord((fila2, col2))
            rodaderos.append((coord_a_celda(x1, y1), coord_a_celda(x2, y2)))

        # Escaleras
        escaleras = []
        for _ in range(m):
            fila1, col1, fila2, col2 = map(int, f.readline().split())
            x1, y1 = cambio_coord((fila1, col1))
            x2, y2 = cambio_coord((fila2, col2))
            escaleras.append((coord_a_celda(x1, y1), coord_a_celda(x2, y2)))

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

    return escaleras, rodaderos, perdida, victoria
