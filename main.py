# Librerías importadas
import matplotlib.pyplot as plt
import numpy as np
# Módulos del proyecto
from tablero import Tablero
from agente import AgenteQLearning
from simulacion import run
from helpers import leer_tablero, guardar_Qtabla, cargar_Qtabla

## Tablero para el proyecto ===============================================
rodaderos, escaleras, perdida, victoria = leer_tablero('./tablero')
tab = Tablero(
    # Características espaciales del tablero
    nro_filas=10,
    nro_columnas=10,
    celdas_victoria=victoria,
    celdas_perdida=perdida,
    celdas_escalera=escaleras,
    celdas_rodadero=rodaderos,
    # Reward
    r_victoria=100, r_perdida=-100, r_otros=-1
)
agente = AgenteQLearning(tab, alpha=0.5, epsilon=0.5, gamma=1)

# Run
reward_historico, Qtabla = run(
    tab, agente, episodios=20000,
    epsilon_ciclos=5, # Número de ciclos de oscilación para epsilon
    print_Qvalores_politica=True, animacion=False,
)
guardar_Qtabla("Qtabla.npy", Qtabla)

# Bootstrapping con un segundo agente =====================================
print("\nBootstrapping:")
Qtabla_bootstrap = cargar_Qtabla("Qtabla.npy")
agente2 = AgenteQLearning(tab, alpha=0.5, epsilon=0, gamma=1)
agente2.Qtabla = Qtabla_bootstrap
_, _ = run(
    tab, agente2, episodios=1,
    print_Qvalores_politica=True, animacion=False
)
