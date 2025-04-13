from typing import Tuple, List  # Para anotar los tipos de argumentos del tablero


def limites_validos(celda, celda_max) -> bool:
    """Validar que el número de celda esté dentro de los límites del tablero"""
    return 1 <= celda <= celda_max


class Tablero:
    def __init__(
        self,
        nro_filas: int,
        nro_columnas: int,
        celda_victoria: int,
        celdas_perdida: List[int],
        celdas_escalera: List[Tuple[int]],
        celdas_rodadero: List[Tuple[int]],
    ):
        """Inicializar y validar un Tablero"""
        if nro_filas < 2 or nro_columnas < 2:
            raise ValueError("Debe haber al menos 2 filas y 2 columnas")
        self.celda_max = nro_filas * nro_columnas

        celdas_a_validar = (
            [celda_victoria]
            + celdas_perdida
            + [celda for par in celdas_escalera for celda in par]
            + [celda for par in celdas_rodadero for celda in par]
        )
        for celda in celdas_a_validar:
            if not limites_validos(celda, self.celda_max):
                raise ValueError(
                    f"{celda} fuera de los limites: {1} a {self.celda_max}"
                )

        self.nro_filas = nro_filas
        self.nro_columnas = nro_columnas
        self.celda_victoria = celda_victoria
        self.celdas_perdida = [c for c in celdas_perdida if c != celda_victoria]
        self.celdas_escalera = celdas_escalera
        self.celdas_rodadero = celdas_rodadero

    def __repr__(self) -> str:
        """Representación impresa del Tablero"""
        atributos = [
            ("Número de filas", self.nro_filas),
            ("Número de columnas", self.nro_columnas),
            ("Celda máxima", self.celda_max),
            ("Celda victoria", self.celda_victoria),
            ("Celdas perdida", self.celdas_perdida),
            ("Celdas escalera", self.celdas_escalera),
            ("Celdas rodadero", self.celdas_rodadero),
        ]
        atributos_str = "\n".join(f" {nombre}: {valor}" for nombre, valor in atributos)
        return f"Tablero:\n{atributos_str}\n"

    def celda_a_coord(self, nro_celda) -> tuple[int, int]:
        """Obtener coordenada (columna, fila) a partir del número de celda y el número de columnas."""

        # -1 para manejar los multiplos de 'nro_columnas'
        fila = ((nro_celda - 1) // self.nro_columnas) + 1
        offset = (nro_celda - 1) % self.nro_columnas

        if fila % 2 == 1:  # Fila impar
            columna = offset + 1  # Izquierda a derecha
        else:
            columna = self.nro_columnas - offset  # Derecha a izquierda
        return (columna - 0.5, fila - 0.5)  # Centrar

    def coord_a_celda(self, col, fila) -> int:
        """Obtener el número de celda a partir del par: (columna, fila)"""
        valor_inicial = (fila - (fila % 2)) * self.nro_columnas
        offset = (-1) ** (fila - 1) * (col - ((fila - 1) % 2))
        return valor_inicial + offset


# Para exportar/importar con `from Tablero import *``
__all__ = ["limites_validos", "Tablero"]
