import random
from tablero import Tablero
import numpy as np


class AgenteQLearning:
    def __init__(self, tablero: Tablero, alpha: float = 0.5, epsilon: float = 0.65, gamma: float = 0.9):

        # Ambiente para el agente
        self.tablero = tablero

        # Parámetros del agente
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Ratio de exploración
        self.pos = 1 # Inicialización de la posición
        # Mapa estado-acciones
        self.s_acciones = {}
        for s in self.tablero.espacio_estados:
            # Al caer en una escalera o rodadero, hay una única acción automática
            if self.tablero.escaleras_y_rodaderos.get(s, False):
                self.s_acciones[s] = ["auto"]
            # No hay acción en celdas terminales
            elif s == tablero.celdas_victoria or s in tablero.celdas_perdida:
                self.s_acciones[s] = [None]
            # Todas las otras celdas tienen dos posibles acciones
            else:
                self.s_acciones[s] = [-1, 1]

        # Q-table: (estado, acción) -> Q
        self.Qtabla = {}

    def escoger_accion(self, estado) -> int:
        """
        Escoge una acción válida para el estado actual con el método epsilon-greedy
        """
        acciones_set = self.s_acciones[estado]

        # Exploración
        if random.random() < self.epsilon:
            return random.choice(acciones_set)

        # Explotación
        else:
            # Obtener los mejores valores Q (0 en caso aún no exista)
            # Usar 0 por defecto hace que el agente prefiera explorar
            Q_vals = [self.Qtabla.get((estado, a), 0) for a in acciones_set]
            Q_max = max(Q_vals)

            # En caso de empate, escoger aleatoriamente una acción
            mejores_acciones = [a for a, Q in zip(acciones_set, Q_vals) if Q == Q_max]
            return random.choice(mejores_acciones)

    def actualizar_Q(self, estado, accion, reward, estado_siguiente) -> None:
        """
        Actualización Q-learning con base en la tupla (Sₜ,Aₜ,Rₜ,Sₜ₊₁):
            Q(Sₜ,Aₜ) = Q(Sₜ,Aₜ) + α[Rₜ₊₁ + γ*maxₐ Q(Sₜ₊₁,a) - Q(Sₜ,Aₜ)]    (Sutton & Barto, p. 131)
        """
        # Q actual y mejor Q en el siguiente estado
        Q_actual = self.Qtabla.get((estado, accion), 0.0)
        Q_siguiente_max = max([self.Qtabla.get((estado_siguiente, a), 0.0) for a in self.s_acciones[estado_siguiente]])

        # Actualización
        self.Qtabla[(estado, accion)] = Q_actual + self.alpha * (reward + self.gamma * Q_siguiente_max - Q_actual)

    def max_Q_values(self):
        """
        Calcular el Q-valor máximo de todos los estados en la Q-table
        """
        estados = {s for (s, _) in self.Qtabla.keys()}
        return {
            s: max(self.Qtabla.get((s, a), 0) for a in self.s_acciones[s])
            for s in estados
        }

    def obtener_Qmax_politica(self):
        filas, columnas = self.tablero.nro_filas, self.tablero.nro_columnas
        Qmax = np.full((filas, columnas), np.nan)
        politica_optima = np.full((filas, columnas), " ")
        auto_char = 'X'
        acciones_str_impar = {-1: '←', 1: '→', 'auto': auto_char}
        acciones_str_par = {-1: '→', 1: '←', 'auto': auto_char}

        # Iterar sobre todos los estados visitados
        estados = {s for (s, _) in self.Qtabla.keys()}
        for s in estados:
            # Índices
            x_int, y_int = self.tablero.celda_a_coord(s, centrar=False)
            i, j = x_int - 1, y_int - 1

            # Mejores acción y q en s
            qvals = {a: self.Qtabla.get((s, a), 0.0) for a in self.s_acciones[s]}
            mejor_a = max(qvals, key=qvals.get) # Equivalente a: argmaxₐ qvals[a]
            mejor_q = self.Qtabla.get((s, mejor_a), 0.0)

            # Poblar arrays
            Qmax[i, j] = np.round(mejor_q, decimals=1)

            if y_int % 2 == 1:
                politica_optima[i, j] = acciones_str_impar[mejor_a]
            else:
                politica_optima[i, j] = acciones_str_par[mejor_a]

        # Rotación para mostrar adecuadamente
        Qmax = np.rot90(Qmax)
        politica_optima = np.rot90(politica_optima)
        return Qmax, politica_optima
