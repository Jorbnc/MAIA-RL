import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def plot_tablero(tablero, trayectoria=None) -> None:
    """
    Plotear el tablero.
    Si se pasa una trayectoria, se mostrará una animación con el recorrido del agente.
    """
    nro_filas, nro_columnas = tablero.nro_filas, tablero.nro_columnas

    fig, ax = plt.subplots(figsize=(nro_columnas, nro_filas), dpi=75)
    ax.set_xlim(0, nro_columnas)
    ax.set_ylim(0, nro_filas)
    ax.set_xticks(range(1, nro_columnas + 1))
    ax.set_yticks(range(1, nro_filas + 1))
    ax.grid(color="black", linewidth=1)
    ax.tick_params(
        axis="both", which="both",
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False,
    )

    # Anotar el número de celda
    for fila in range(1, nro_filas + 1):
        for col in range(1, nro_columnas + 1):
            nro_celda = tablero.coord_a_celda(col, fila)
            ax.text(col - 0.5, fila - 0.35, str(nro_celda), ha="center", va="bottom", alpha=0.5)

    # Posición de victoria
    vic_coord = tablero.celda_a_coord(tablero.celda_victoria)
    ax.text(*vic_coord, "FIN", ha="center", va="top", color="green", size=15)

    # Posiciones de pérdida
    for celda in tablero.celdas_perdida:
        ax.scatter(*tablero.celda_a_coord(celda), color="red", s=150, marker="X")

    # Escaleras
    for par in tablero.celdas_escalera:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.08, length_includes_head=True,
                                    color="green", alpha=0.5)
        ax.add_patch(arrow)

    # Rodaderos
    for par in tablero.celdas_rodadero:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.08, length_includes_head=True,
                                    color="red", alpha=0.5)
        ax.add_patch(arrow)

    # Animar trayectoria
    if trayectoria:
        dot, = ax.plot([], [], 'bo', markersize=12) # Figura del agente. Se actualiza iterativamente

        def actualizar_frame(frame):
            coord = tablero.celda_a_coord(trayectoria[frame])
            dot.set_data([coord[0]], [coord[1]]) # set_data espera un array, no valores escalares
            return dot, # tuple (con 1 solo elemento) para que FuncAnimation pueda 'iterar' y re-dibujar el dot

        anim = animation.FuncAnimation( # Es necesario asignar a una variable para que el GC no lo elimine
            fig,
            actualizar_frame, # Función para llamar por cada frame
            frames=len(trayectoria), # Total de frames
            interval=200, # milisegundos
            repeat=True, # loopear la animación
            blit=True, # re-dibujar solo figuras que han cambiado
        )

    plt.show()
