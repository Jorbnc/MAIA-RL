import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def plot_tablero(tablero, trayectoria=None, Q_values_lista=None) -> None:
    """
    Plotear el tablero.
    Si se pasa una trayectoria, se mostrará una animación con el recorrido del agente.
    """
    nro_filas, nro_columnas = tablero.nro_filas, tablero.nro_columnas

    fig, (ax, cb) = plt.subplots(
        1, 2, figsize=(nro_columnas, nro_filas),
        dpi=75, gridspec_kw={'width_ratios': [15, 1]}
    )
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
            ax.text(col - 0.5, fila - 0.35, str(nro_celda), ha="center", va="bottom", alpha=1)

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

    # Plotear Q-values
    if Q_values_lista:

        # Generar array con Q_values
        def Q_values_map(Q_dict):
            vals = np.zeros((nro_filas, nro_columnas))
            for celda, q in Q_dict.items():
                x, y = tablero.celda_a_coord(celda)
                vals[int(y), int(x)] = q
            return vals

        vmaps = [Q_values_map(qvals) for qvals in Q_values_lista]

        # Norma divergente inicial para el colormap, centrada en 0
        max_abs = np.max(np.abs(vmaps[0]))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        # Heatmap y Colorbar iniciales
        im = ax.imshow(
            vmaps[0],
            origin='lower',
            extent=[0, nro_columnas, 0, nro_filas],
            cmap='PiYG',
            norm=norm,
            alpha=0.5
        )
        cbar = plt.colorbar(im, cb)

        # Animación
        def actualizar_frame(f):
            # Actualizar vmap
            vmap = vmaps[f]
            im.set_data(vmap)

            # recompute dynamic range and update norm
            lo, hi = vmap.min(), vmap.max()
            im.set_clim(lo, hi)
            cbar.update_normal(im)
            ax.set_title(f"Episodio {f}", fontsize=15)

            return im, cbar

        anim_1 = animation.FuncAnimation(
            fig, actualizar_frame, frames=len(vmaps),
            interval=100,
            repeat=False,
            blit=False, # redibujar todo
        )
        # plt.show()

    # Animar trayectoria
    if trayectoria:
        dot, = ax.plot([], [], 'bo', markersize=12) # Figura del agente. Se actualiza iterativamente

        def actualizar_frame(frame):
            coord = tablero.celda_a_coord(trayectoria[frame])
            dot.set_data([coord[0]], [coord[1]]) # set_data espera un array, no valores escalares
            return dot, # tuple (con 1 solo elemento) para que FuncAnimation pueda 'iterar' y re-dibujar el dot

        anim_2 = animation.FuncAnimation( # Es necesario asignar a una variable para que el GC no lo elimine
            fig,
            actualizar_frame, # Función para llamar por cada frame
            frames=len(trayectoria), # Total de frames
            interval=100, # milisegundos
            repeat=False, # loopear la animación
            blit=False, # re-dibujar solo figuras que han cambiado
        )

    plt.show()
