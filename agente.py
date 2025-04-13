import random
from typing import Tuple, List
from tablero import Tablero


class Agente_QLearning:
    def __init__(self, tablero: Tablero, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):

        # Ambiente
        self.tablero = tablero

        # Parámetros del agente
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Ratio de exploración
        self.pos = 1
        self.acciones = [-1, 1]

        self.escaleras_a = [a for (a, b) in tablero.celdas_escalera]
        self.escaleras_b = [b for (a, b) in tablero.celdas_escalera]
        self.rodaderos_a = [a for (a, b) in tablero.celdas_rodadero]
        self.rodaderos_b = [b for (a, b) in tablero.celdas_rodadero]

        # Initialize Q-table as a dictionary keyed by (state, action)
        self.Q = {}  # type: dict[Tuple[int, int], float]

    def escoger_accion(self, estado) -> int:
        """Escoger una acción con una política epsilon-greedy"""

        # Exploración
        if random.random() < self.epsilon:
            return random.choice(self.acciones)
        # Explotación
        else:
            # Get Q-values for available actions (default 0 if not present)
            q_vals = [self.Q.get((estado, a), 0.0) for a in self.acciones]
            q_max = max(q_vals)
            # In case of tie, randomly choose among best actions.
            best_actions = [a for a, q in zip(self.acciones, q_vals) if q == q_max]
            return random.choice(best_actions)

    def update(self, estado, accion, reward, estado_siguiente) -> None:
        """Actualizar el Q-value"""
        q_actual = self.Q.get((estado, accion), 0.0)

        # Estimate the best future value at next_state.
        q_vals_siguiente = [self.Q.get((estado_siguiente, a), 0.0) for a in self.acciones]
        q_siguiente_max = max(q_vals_siguiente)

        # Q-learning update
        q_nuevo = q_actual + self.alpha * (reward + self.gamma * q_siguiente_max - q_actual)
        self.Q[(estado, accion)] = q_nuevo

    def step(self) -> Tuple[int, int, float, int]:
        """Retorna: (estado, accion, reward, estado_siguiente)"""

        estado = self.pos
        accion = self.escoger_accion(estado)

        # Transición al siguiente estado dentro de los límites
        estado_siguiente = estado + accion
        estado_siguiente = max(1, min(estado_siguiente, self.tablero.celda_max))

        # Rewards
        if estado_siguiente == self.tablero.celda_victoria:
            reward = 1.0
        elif estado_siguiente in self.tablero.celdas_perdida:
            reward = -1.0
        else:
            reward = 0.0

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

        # Actualizar Q-learning y estado (pos)
        self.update(estado, accion, reward, estado_siguiente)
        self.pos = estado_siguiente

        return estado, accion, reward, estado_siguiente
