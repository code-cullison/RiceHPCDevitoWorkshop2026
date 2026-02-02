import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from devito import Function, Eq, Operator
from devito.symbolics import retrieve_functions


def plot_equation_grid(equations, target_func, grid, ncols=3, cmap_name="jet"):
    """
    Plot each equation in a 2D grid of subplots (3 columns, as many rows as needed).
    Args:
        equations: List of equations to plot.
        target_func: The Function to update and plot.
        grid: The computational grid (for extent).
        ncols: Number of columns in the subplot grid (default 3).
        cmap_name: Name of the matplotlib colormap (default: 'jet').
    """

    n_eqs = len(equations)
    nrows = int(np.ceil(n_eqs / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    print(f'axes.shape B4: {axes.shape}')
    axes = np.array(axes).reshape(-1)
    print(f'axes.shape AF: {axes.shape}')

    for i, (eq, ax) in enumerate(zip(equations, axes)):
        target_func.data[:] = 0
        Operator(eq)()

        # setup colormap and normalization
        dmax = max(np.max(target_func.data),len(equations))
        dmin = np.min(target_func.data)
        print(f'target_func.data min/max: {dmin}/{dmax}')
        assert dmin >= 0, "Data contains negative values."
        levels = np.arange(-0.5, dmax + 0.5 + 1, 1)
        print(f'levels: {levels}')
        cmap = plt.get_cmap(cmap_name, len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)

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
        cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0, int(dmax)+1), spacing="proportional")
        cbar.set_label("Amplitude")

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    return fig, axes



def create_domain_functions_and_equations(grid, subdomains):
    """
    Given a grid and a list of subdomains, create a Function for each subdomain,
    plus a default 'full_f' for the full grid. Then create equations assigning
    unique integer values to each subdomain's function, and return a list of
    combined equations and the full domain function.
    """
    # Create the function for the full domain
    full_f = Function(name='full_f', grid=grid)
    eq_full = Eq(full_f, 0, subdomain=grid)
    
    # Create functions and equations for each subdomain
    subdomain_functions = []
    subdomain_equations = []
    for idx, subd in enumerate(subdomains, start=1):
        # Use the subdomain's name attribute for the function name
        subd_name = getattr(subd, 'name', f'subd{idx}')
        func = Function(name=f'{subd_name}_f', grid=grid)
        eq = Eq(func, idx, subdomain=subd)   # func = idx in this subdomain (integer value)
        subdomain_functions.append(func)
        subdomain_equations.append(eq)
    
    # Combine all equations: full domain + subdomains
    all_equations = [eq_full] + subdomain_equations

    # Run the operator to apply the each equation separately
    _ = Operator(all_equations)()
    
    # For demonstration, create combined equations (step-by-step inclusion of subdomains) 
    import operator
    from itertools import accumulate
    lhs = full_f
    term_list = [eq.lhs for eq in all_equations]
    partial_sums = list(accumulate(term_list[1:], operator.add))
    rhs_list = [lhs + s for s in partial_sums]
    combined_equations = [Eq(lhs, rhs) for rhs in rhs_list]
    
    # Run the operator for the combined equations
    _ = Operator(combined_equations)()
    
    return combined_equations, full_f