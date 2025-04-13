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
        self.pos = 1
        self.acciones = [-1, 1] # Izquierda/Derecha o Arriba/Abajo en los bordes del tablero

        # Algunas asignanciones locales para no generar estas listas por cada iteración
        self.escaleras_a = [a for (a, b) in tablero.celdas_escalera]
        self.escaleras_b = [b for (a, b) in tablero.celdas_escalera]
        self.rodaderos_a = [a for (a, b) in tablero.celdas_rodadero]
        self.rodaderos_b = [b for (a, b) in tablero.celdas_rodadero]

        # Q-table: Es un mapeo (estado, acción) -> Q estimado (actualizado iterativamente)
        self.Q = {}  # dict[Tuple[int, int], float]

    def escoger_accion(self, estado) -> int:
        """Escoger una acción con el método epsilon-greedy"""

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
        Actualización Q-learning con base en (s,a,r,s'):

            Q(Sₜ,Aₜ) = Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ*maxₐ Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]    (Sutton & Barto, p. 131)
        """
        Q_actual = self.Q.get((estado, accion), 0.0)

        # Estimar mejor Q en el siguiente estado
        Q_vals_siguiente_max = max([self.Q.get((estado_siguiente, a), 0.0) for a in self.acciones])

        # Actualización
        self.Q[(estado, accion)] = Q_actual + self.alpha * (reward + self.gamma * Q_vals_siguiente_max - Q_actual)

    def step(self) -> Tuple[int, int, float, int]:
        """
        Representa un paso en la simulación con base en las condiciones actuales:
            - Implementa la transición entre estados y su validación
            - Calcula las recompensas
            - Actualiza Q
        Retorna: (s,a,r,s')
        """

        estado = self.pos
        accion = self.escoger_accion(estado)

        # Transición al siguiente estado, asegurando los límites
        estado_siguiente = estado + accion
        estado_siguiente = max(1, min(estado_siguiente, self.tablero.celda_max))

        # Transición para Escaleras y Rodaderos
        try:
            idx = self.escaleras_a.index(estado_siguiente)
            estado_siguiente = self.escaleras_b[idx]
        except ValueError:
            try:
                idx = self.rodaderos_a.index(estado_siguiente)
                estado_siguiente = self.rodaderos_b[idx]
            except ValueError:
                pass

        # Rewards
        if estado_siguiente == self.tablero.celda_victoria:
            reward = 1.0
        elif estado_siguiente in self.tablero.celdas_perdida:
            reward = -1.0
        else:
            reward = 0.0

        # Actualizar Q y estado (pos)
        self.actualizar_Q(estado, accion, reward, estado_siguiente)
        self.pos = estado_siguiente

        return estado, accion, reward, estado_siguiente
