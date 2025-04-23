import random
from typing import Tuple, List
from tablero import Tablero


class AgenteQLearning:
    def __init__(self, tablero: Tablero, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):

        # Ambiente
        self.tablero = tablero

        # Parámetros del agente
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Ratio de exploración
        self.pos = 1 # Inicialización de la posición
        self.acciones = [-1, 1] # Izquierda/Derecha o Arriba/Abajo en los bordes del tablero

        # Q-table (o Q-dict en nuestro caso)
        # Es un mapeo (estado, acción) -> Q estimado (actualizado iterativamente)
        self.Q = {}

    def escoger_accion(self, estado) -> int:
        """Escoge una acción con el método epsilon-greedy"""

        # Exploración
        if random.random() < self.epsilon:
            return random.choice(self.acciones)

        # Explotación
        else:
            # Obtener los mejores valores Q (0 en caso aún no exista)
            Q_vals = [self.Q.get((estado, a), 0.0) for a in self.acciones]
            Q_max = max(Q_vals)

            # En caso de empate, escoger aleatoriamente una acción
            mejores_acciones = [a for a, Q in zip(self.acciones, Q_vals) if Q == Q_max]
            return random.choice(mejores_acciones)

    def actualizar_Q(self, estado, accion, reward, estado_siguiente) -> None:
        """
        Actualización Q-learning con base en(Sₜ,Aₜ,Rₜ,Sₜ₊₁):
            Q(Sₜ,Aₜ) = Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ*maxₐ Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]    (Sutton & Barto, p. 131)
        """
        Q_actual = self.Q.get((estado, accion), 0.0)

        # Estimar mejor Q en el siguiente estado
        Q_vals_siguiente_max = max([self.Q.get((estado_siguiente, a), 0.0) for a in self.acciones])

        # Actualización
        # TODO: Probar si hay cambios importantes al variar gamma
        self.Q[(estado, accion)] = Q_actual + self.alpha * (reward + self.gamma * Q_vals_siguiente_max - Q_actual)

    def transicion(self, estado, accion):
        """
        Función de transición que retorna una celda entre dos posibles opciones:
            - El final de una escalera/rodadero
            - Otra celda válida dentro del tablero
        """
        estado_siguiente = estado + accion
        estado_siguiente = self.tablero.escaleras_y_rodaderos.get(
            # Si el estado_siguiente es el inicio de una escalera/rodadero, entonces retorna el final
            estado_siguiente,
            # En caso contrario, solo valida el estado_siguiente
            max(1, min(estado_siguiente, self.tablero.celda_max))
        )
        # WARNING: No se llega a evaluar una condición de finalización aquí
        # ya que eso se está manejando externamente en simulacion.py
        return estado_siguiente

    def reward(self, estado_siguiente):
        """
        Función de recompensa basado en un mapeo (diccionario): Sₜ₊₁ -> reward
            - +1 para celda victoria
            - -1 para celdas perdida
            - 0 para los otros casos
        """
        return self.tablero.reward_map.get(estado_siguiente, 0)

    def step(self) -> Tuple[int, int, float, int]:
        """
        Representa un paso en la simulación con base en las condiciones actuales.
        Actualiza la Q-table con base en (Sₜ,Aₜ,Rₜ,Sₜ₊₁) y retorna la tupla.
        """

        estado = self.pos
        accion = self.escoger_accion(estado)

        # Transición y Reward del paso actual
        estado_siguiente = self.transicion(estado, accion)
        reward = self.reward(estado_siguiente)

        # Actualizar Q y estado (pos)
        self.actualizar_Q(estado, accion, reward, estado_siguiente)
        self.pos = estado_siguiente

        return estado, accion, reward, estado_siguiente
