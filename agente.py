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

        # Definiciones internas para efectuar búsquedas eficientes
        E_inicio = [a for (a, b) in tablero.celdas_escalera] + [a for (a, b) in tablero.celdas_rodadero]
        E_fin = [b for (a, b) in tablero.celdas_escalera] + [b for (a, b) in tablero.celdas_rodadero]
        self.E_dict = {inicio: fin for inicio, fin in zip(E_inicio, E_fin)}

        self.reward_map = {
            self.tablero.celda_victoria: 1.0,
            **{cell: -1.0 for cell in self.tablero.celdas_perdida}
        }

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

    def transicion(self, estado, accion):
        """
        Función de transición que retorna:
            - El final de una escalera/rodadero
            - U otra celda válida dentro del tablero
        """
        estado_siguiente = estado + accion
        estado_siguiente = self.E_dict.get(
            estado_siguiente, # f(Sₜ,Aₜ)
            max(1, min(estado_siguiente, self.tablero.celda_max))
        )
        return estado_siguiente

    def reward(self, estado_siguiente):
        return self.reward_map.get(estado_siguiente, 0)

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

        # Transición y Reward del paso actual
        estado_siguiente = self.transicion(estado, accion)
        reward = self.reward(estado_siguiente)

        # Actualizar Q y estado (pos)
        self.actualizar_Q(estado, accion, reward, estado_siguiente)
        self.pos = estado_siguiente

        return estado, accion, reward, estado_siguiente
