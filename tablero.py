from typing import Tuple, List


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
        # Validación de celdas ----------------------------------------------------------
        if nro_filas < 2 or nro_columnas < 2:
            raise ValueError("Debe haber al menos 2 filas y 2 columnas")
        self.celda_max = nro_filas * nro_columnas
        celdas_a_validar = ([celda_victoria] + celdas_perdida
                            + [celda for par in celdas_escalera for celda in par]
                            + [celda for par in celdas_rodadero for celda in par]
                            )
        # Todas los números de celda deben encontrarse dentro de los límites
        for celda in celdas_a_validar:
            if not (1 <= celda <= self.celda_max):
                raise ValueError(f"{celda} fuera de los limites: {1} a {self.celda_max}")
        # -------------------------------------------------------------------------------

        # Atributos
        self.nro_filas = nro_filas
        self.nro_columnas = nro_columnas
        self.celda_victoria = celda_victoria
        self.celdas_perdida = [c for c in celdas_perdida if c != celda_victoria]
        self.celdas_escalera = celdas_escalera
        self.celdas_rodadero = celdas_rodadero
        self.espacio_estados = range(1, nro_filas * nro_columnas + 1)

        # Ya que la transición desde el inicio hacia el final de una escalera tiene la misma
        # lógica que la de un rodadero, podemos manejar este movimiento con un solo diccionario
        self.escaleras_y_rodaderos = {
            **dict(self.celdas_escalera),
            **dict(self.celdas_rodadero)
        }

        # Reward para los estados terminales. Esta estructura se define fuera del método
        # reward para no tener que construir el diccionario cada vez que se llame a dicha función
        self.reward_terminales = {
            self.celda_victoria: 50.0,
            **{cell: -50.0 for cell in self.celdas_perdida}
        }

    # Métodos auxiliares para manejar las equivalencias 2D <-> 1D -------------------
    def celda_a_coord(self, nro_celda, centrar=True) -> tuple[float, float]:
        """
        Obtener coordenada (columna, fila) a partir del número de celda.
        """
        # -1 para manejar los multiplos de 'nro_columnas'
        fila = ((nro_celda - 1) // self.nro_columnas) + 1
        offset = (nro_celda - 1) % self.nro_columnas

        # Si la fila es impar...
        if fila % 2 == 1:
            columna = offset + 1  # ...la secuencia va de izquierda a derecha
        # Si la fila es par...
        else:
            columna = self.nro_columnas - offset  # ...la secuencia va de derecha a izquierda
        # Retornar coordenada centrada (para ploteo principalmente)
        if centrar:
            return (columna - 0.5, fila - 0.5)
        # O como valores enteros sin centrar
        return (columna, fila)

    def coord_a_celda(self, col, fila) -> int:
        """
        Obtener el número de celda a partir del par: (columna, fila).
        """
        # Valor base:
        #   Coordenadas con filas pares inician su conteo en fila * nro_columnas
        #   Coordenadas con filas impares inician su conteo en (fila - 1) * nro_columnas
        valor_inicial = (fila - (fila % 2)) * self.nro_columnas
        # Valor offset:
        #   Coordenadas con filas pares disminuyen de izquierda a derecha
        #   Coordenadas con filas impares aumentan de izquierda a derecha
        offset = (-1) ** (fila - 1) * (col - ((fila - 1) % 2))
        # Sumar y retornar ambos valores
        return valor_inicial + offset

    # Métodos del MDP ----------------------------------------------------------------
    def transicion(self, estado, accion):
        """
        Función de transición que retorna una celda entre dos posibles opciones:
            - El final de una escalera/rodadero
            - Celda adyacente válida dentro de los límites
        """
        # Si el estado actual es el inicio de una escalera/rodadero, entonces retorna el final
        if estado in self.escaleras_y_rodaderos.keys():
            return self.escaleras_y_rodaderos[estado]

        # En caso contrario, solo valida límites
        else:
            return max(1, min(estado + accion, self.celda_max))
            # Nota: La condición de finalización se evalúa en simulacion.py

    def reward(self, estado_siguiente):
        """
        Función de recompensa para Sₜ₊₁ con dos posibles salidas:
            - La recompensa de los estados terminales
            - Valor por defecto para todos los otros estados
        """
        return self.reward_terminales.get(
            # Recompensa de celda victoria o de celda trampa
            estado_siguiente,
            # Recompensa negativa. La intención es presionar al agente para ganar lo antes posible
            -1
        )
