import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import get_cmap
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
        ax.text(*vic_coord, "FIN", ha="center", va="top", color="blue", size=15)

    # Posiciones de pérdida
    for celda in tablero.celdas_perdida:
        ax.scatter(*tablero.celda_a_coord(celda), color="red", s=150, marker="X")

    # Escaleras
    for par in tablero.celdas_escalera:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.08, length_includes_head=True,
                                    color="blue", alpha=0.5)
        ax.add_patch(arrow)

    # Rodaderos
    for par in tablero.celdas_rodadero:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.08, length_includes_head=True,
                                    color="black", alpha=0.5)
        ax.add_patch(arrow)

    # Animar evolución de Q-values y trayectoria
    if Q_values_lista and trayectoria:

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
            # cmap='PRGn',
            cmap=get_cmap('seismic_r'),
            norm=norm,
            alpha=0.5
        )
        cbar = plt.colorbar(im, cb)

        # Figura del agente
        dot, = ax.plot([], [], 'bo', markersize=12)

        num_episodios = len(vmaps)
        frames_totales = num_episodios + len(trayectoria)

        # Animación
        def actualizar_frame(f):
            artists = [] #

            if f < num_episodios:
                # Actualizar vmap
                vmap = vmaps[f]
                im.set_data(vmap)

                # recompute dynamic range and update norm
                lo, hi = vmap.min(), vmap.max()
                im.set_clim(lo, hi)
                cbar.update_normal(im)
                ax.set_title(f"Episodio {f}", fontsize=15)
                artists += [im, cbar]

            else:
                ax.set_title(f"Episodio {num_episodios}", fontsize=15)
                idx = f - num_episodios
                x, y = tablero.celda_a_coord(trayectoria[idx])
                dot.set_data([x], [y]) # set_data espera un array, no valores escalares
                artists.append(dot)

            return artists

        anim = animation.FuncAnimation(
            fig, actualizar_frame, frames=frames_totales,
            interval=125,
            repeat=True,
            blit=False, # no redibujar todo
        )

    plt.show()
