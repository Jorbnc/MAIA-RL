import random
from tablero import Tablero


class AgenteQLearning:
    def __init__(self, tablero: Tablero, alpha: float = 0.25, gamma: float = 0.9, epsilon: float = 0.25):

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

    # FIX:
    def accionar(self, estado):
        return

    def actualizar_Q(self, estado, accion, reward, estado_siguiente) -> None:
        """
        Actualización Q-learning con base en(Sₜ,Aₜ,Rₜ,Sₜ₊₁):
            Q(Sₜ,Aₜ) = Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ*maxₐ Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]    (Sutton & Barto, p. 131)
        """
        Q_actual = self.Q.get((estado, accion), 0.0)

        # Estimar mejor Q en el siguiente estado
        Q_vals_siguiente_max = max([self.Q.get((estado_siguiente, a), 0.0) for a in self.acciones])

        # Actualización
        self.Q[(estado, accion)] = Q_actual + self.alpha * (reward + self.gamma * Q_vals_siguiente_max - Q_actual)

    def Q_values(self):
        estados = {s for (s, a) in self.Q.keys()}
        p = 1 / len(self.acciones)
        return {s: sum(p * self.Q.get((s, a), 0) for a in self.acciones) for s in estados}
