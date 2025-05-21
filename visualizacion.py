import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import get_cmap
import numpy as np
from tablero import coord_a_celda


def plot_tablero(tablero, a_params, Q_values_lista, trayectoria, epsilon_episodico, interval) -> None:
    """
    Plotear el tablero y mostrar dos animaciones:
        - Evolución de los estado-valores durante todos los episodios
        - Recorrido del agente con la última política obtenida (episodio final)
    """
    nro_filas, nro_columnas = tablero.nro_filas, tablero.nro_columnas

    # Configuración del plot ---------------------------------------------------------------
    fig = plt.figure(figsize=(nro_columnas, nro_filas), dpi=80)
    gs = GridSpec(2, 2, width_ratios=[15, 1], height_ratios=[4, 1], hspace=0.25)

    t_axis = fig.add_subplot(gs[0, 0])
    cb_axis = fig.add_subplot(gs[0, 1])
    eps_axis = fig.add_subplot(gs[1, :])

    t_axis.set_xlim(0, nro_columnas)
    t_axis.set_ylim(0, nro_filas)
    t_axis.set_xticks(range(1, nro_columnas + 1))
    t_axis.set_yticks(range(1, nro_filas + 1))
    labels_x = [str(i) for i in range(nro_columnas)]
    labels_y = [str(i) for i in range(nro_filas - 1, -1, -1)]
    t_axis.set_xticklabels(labels_x, color='gray')
    t_axis.set_yticklabels(labels_y, color='gray')
    plt.setp(t_axis.xaxis.get_majorticklabels(), ha="right")
    plt.setp(t_axis.yaxis.get_majorticklabels(), va="top")
    t_axis.grid(color="black", linewidth=1)
    t_axis.tick_params(
        axis="both", which="both",
        bottom=True, top=False, left=False, right=False,
        labelbottom=True, labelleft=True,
    )

    # Anotaciones en el tablero ------------------------------------------------------------
    # Números de celdas
    for fila in range(1, nro_filas + 1):
        for col in range(1, nro_columnas + 1):
            nro_celda = coord_a_celda(col, fila, nro_columnas)
            t_axis.text(col - 0.5, fila - 0.35, str(nro_celda), ha="center", va="bottom", alpha=1)

    # Posiciones de victoria
    for celda in tablero.celdas_victoria:
        t_axis.text(*tablero.celda_a_coord(celda), "V", ha="center", va="top", color="blue", size=15)

    # Posiciones de pérdida
    for celda in tablero.celdas_perdida:
        t_axis.scatter(*tablero.celda_a_coord(celda), color="red", s=150, marker="X")

    # Escaleras y Rodaderos ----------------------------------------------------------------
    for par in tablero.celdas_escalera:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.05, length_includes_head=True,
                                    color="blue", alpha=0.5)
        t_axis.add_patch(arrow)

    for par in tablero.celdas_rodadero:
        x1, y1 = tablero.celda_a_coord(par[0])
        x2, y2 = tablero.celda_a_coord(par[1])
        arrow = mpatches.FancyArrow(x1, y1, x2 - x1, y2 - y1,
                                    width=0.05, length_includes_head=True,
                                    color="black", alpha=0.5)
        t_axis.add_patch(arrow)

    # Animar evolución de Q-values y trayectoria -------------------------------------------
    # Lista con los mapas de estado-valores
    vmaps = []
    # Masks para cubrir celdas no visitadas (con un color distinto al colormap principal)
    masks = []
    # Por cada episodio...
    for qdict in Q_values_lista:
        # ... inicializar Q-valores en 0 y todas las celdas como 'no visitadas'
        vmap_vals = np.zeros((nro_filas, nro_columnas))
        mask_novisitada = np.ones((nro_filas, nro_columnas), dtype=bool)
        # Actualizar valores y celda visitadas
        for celda, q in qdict.items():
            # Coordenadas (con el cambio: coord2 = fila, coord1 = columna)
            x, y = tablero.celda_a_coord(celda)
            ix, iy = int(y), int(x)
            # Q-valor actual y celda visitada
            vmap_vals[ix, iy] = q
            mask_novisitada[ix, iy] = False
        vmaps.append(vmap_vals)
        masks.append(mask_novisitada)

    # Norma para el colormap divergente
    max_abs = np.max(np.abs(vmaps[0]))
    # norm = TwoSlopeNorm(vmin=min_abs, vcenter=0.0, vmax=max_abs)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    # Mask inicial de celdas no visitadas
    mask0 = np.ma.MaskedArray(vmaps[0], mask=masks[0])

    # Mapa
    im = t_axis.imshow(
        mask0,
        origin='lower',
        extent=[0, nro_columnas, 0, nro_filas],
        cmap=get_cmap('bwr_r'), norm=norm,
        alpha=0.5
    )

    # Color de celdas no visitadas
    im.cmap.set_bad(color='black', alpha=1)

    # Frames totales para la animación: Entrenamiento + Trayectoria final
    num_episodios = len(vmaps)
    frames_totales = num_episodios + len(trayectoria)

    # Plot inicial de color bar y figura que representa al agente (se actualizarán por cada frame)
    cbar = plt.colorbar(im, cb_axis, label='Q-valor')
    dot, = t_axis.plot([], [], 'bo', markersize=12)
    t_axis.set_xlabel(
        f"(celdas terminales y nunca visitadas en gris)",
        fontsize=13
    )

    # Epsilon por episodio
    eps_path = eps_axis.plot(range(num_episodios + 1), epsilon_episodico, color='black', alpha=0.50)
    dot_eps, = eps_axis.plot([], [], marker=r'$\epsilon$', linestyle='None', color='black', markersize=10)
    eps_axis.set_xlim(0, num_episodios)
    eps_axis.set_ylim(0, 1)
    eps_axis.set_title("Oscilación de epsilon")
    eps_axis.grid(True)

    # Función de actualización de frames

    def actualizar_frame(f):
        #
        artists = []

        # Animación de Q-valores
        if f < num_episodios:
            m = np.ma.MaskedArray(vmaps[f], mask=masks[f])
            im.set_data(m)

            # Actualizar colores y colorbar
            im.set_clim(m.min(), m.max())
            cbar.update_normal(im)
            t_axis.set_title(
                f"Q-valor máximo por estado para el episodio {f}\nα={a_params[0]}, ε={a_params[1]}, γ={a_params[2]}",
                fontsize=15
            )

            # Actualizar valor epsilon
            dot_eps.set_data([f], [epsilon_episodico[f]])

            artists += [im, cbar, dot_eps]

        # Animación del recorrido
        else:
            t_axis.set_title(f"Q-valor máximo por estado para el episodio {num_episodios}", fontsize=17)
            x, y = tablero.celda_a_coord(trayectoria[f - num_episodios])
            dot.set_data([x], [y])
            dot_eps.set_data([f], [epsilon_episodico[-1]])
            artists.append(dot)

        return artists

    # Se tiene que guardar la animación en una variable para que no se elimine con el Garbage Collector
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    anim = animation.FuncAnimation(
        fig,
        actualizar_frame,
        frames=frames_totales,
        interval=interval,
        repeat=True,
        blit=False,
    )

    plt.show()
