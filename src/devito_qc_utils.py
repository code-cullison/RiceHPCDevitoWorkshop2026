import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from devito.symbolics import retrieve_functions
from devito import Operator

def plot_equation_grid(equations, target_func, grid, ncols=3, levels=None, cmap_name="jet"):
    """
    Plot each equation in a 2D grid of subplots (3 columns, as many rows as needed).
    Args:
        equations: List of equations to plot.
        target_func: The Function to update and plot.
        grid: The computational grid (for extent).
        ncols: Number of columns in the subplot grid (default 3).
        levels: Discrete color levels (default: np.arange(-0.5, 5.5+1, 1)).
        cmap_name: Name of the matplotlib colormap (default: 'jet').
    """
    if levels is None:
        levels = np.arange(-0.5, 5.5 + 1, 1)
    cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)

    n_eqs = len(equations)
    nrows = int(np.ceil(n_eqs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).reshape(-1)

    for i, (eq, ax) in enumerate(zip(equations, axes)):
        target_func.data[:] = 0
        Operator(eq)()

        # Build a compact symbolic title: full_f = full_f + main_f + left_f
        try:
            lhs_name = eq.lhs.function.name
        except AttributeError:
            lhs_name = str(eq.lhs)
        rhs_funcs = sorted({f.function.name for f in retrieve_functions(eq.rhs)})
        rhs_label = " + ".join(rhs_funcs)
        title_str = rf"$ {lhs_name} = {rhs_label} $"

        im = ax.imshow(
            target_func.data.T,
            origin="upper",
            extent=[0, grid.extent[0], grid.extent[1], 0],
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title_str, fontsize=11)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0, 6), spacing="proportional")
        cbar.set_label("Amplitude")

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    plt.show()
