import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functools import wraps
from contextlib import contextmanager

from devito import Function, Eq, Operator, configuration
from devito.symbolics import retrieve_functions


#################################################################################
##
## If you want to set the Logging-Level of Devito Operator class when called
##
#################################################################################

def myloglevel_devito(level):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_level = configuration['log-level']
            try:
                configuration['log-level'] = level
                return func(*args, **kwargs)
            finally:
                configuration['log-level'] = old_level
        return wrapper
    return decorator

@myloglevel_devito('ERROR')
def myquiet_op(op, nt=None):
    """Run operator with ERROR log level."""
    if nt is not None:
        return op.apply(time_M=nt)
    else:
        return op()

@myloglevel_devito('DEBUG')
def mydebug_op(op, nt=None):
    """Run operator with DEBUG log level."""
    if nt is not None:
        return op.apply(time_M=nt)
    else:
        return op()


#################################################################################
##
## If you want to set any log level for a blocked code region, including Devito 
## Example: with devito_log_level('ERROR'): <code block>
##
#################################################################################
@contextmanager
def devito_log_level(level):
    old_level = configuration['log-level']
    try:
        configuration['log-level'] = level
        yield
    finally:
        configuration['log-level'] = old_level


#################################################################################
##
## Convert coordinates to indices, with NaN for out-of-bounds
##
#################################################################################
def coordinates_to_indices(coords, grid):
    """
    Convert a single (x, y) coordinate pair to data indices (ix, iy) for a Devito grid.

    If the coordinate is outside the grid extent, returns -np.nan if it is left or above
    the grid, and +np.nan if it is right or below the grid, for each dimension.

    Parameters
    ----------
    coords : iterable of float
        A single (x, y) coordinate pair.
    grid : devito.Grid
        The Devito grid object.

    Returns
    -------
    tuple
        (ix, iy) data indices as integers if inside the grid, or -np.nan/+np.nan if out of bounds.
    """
    x, y = coords
    ox, oy = grid.origin
    sx, sy = grid.spacing
    nx, ny = grid.shape

    # Compute index for x
    if x < ox:
        ix = -np.nan
    elif x > ox + sx * (nx - 1):
        ix = +np.nan
    else:
        ix = int(round((x - ox) / sx))

    # Compute index for y
    if y < oy:
        iy = -np.nan
    elif y > oy + sy * (ny - 1):
        iy = +np.nan
    else:
        iy = int(round((y - oy) / sy))

    return (ix, iy)


#################################################################################
##
## Convert indices to coordinates, with NaN for out-of-bounds
##
#################################################################################
def indices_to_coordinates(indices, grid):
    """
    Convert a single (ix, iy) data index pair to physical coordinates (x, y) for a Devito grid.

    If the index is outside the grid shape, returns -np.nan if it is left or above
    the grid, and +np.nan if it is right or below the grid, for each dimension.

    Parameters
    ----------
    indices : iterable of int
        A single (ix, iy) data index pair.
    grid : devito.Grid
        The Devito grid object.

    Returns
    -------
    tuple
        (x, y) physical coordinates as floats if inside the grid, or -np.nan/+np.nan if out of bounds.
    """
    ix, iy = indices
    ox, oy = grid.origin
    sx, sy = grid.spacing
    nx, ny = grid.shape

    # Compute x coordinate
    if ix < 0:
        x = -np.nan
    elif ix > nx - 1:
        x = +np.nan
    else:
        x = ox + ix * sx

    # Compute y coordinate
    if iy < 0:
        y = -np.nan
    elif iy > ny - 1:
        y = +np.nan
    else:
        y = oy + iy * sy

    return (x, y)


#################################################################################
##
## Get stencil center coordinates one half-width away from each corner
##
#################################################################################
def get_near_corner_points(grid, so):
    """
    Return grid indices and physical coordinates for points near each corner of a 2D grid.

    The points are offset from each corner by half the stencil width (so//2), which is useful
    for placing sources, receivers, or boundary conditions away from the very edge of the domain.

    Parameters
    ----------
    grid : devito.Grid
        The computational grid object (must be 2D).
    so : int
        The spatial finite-difference stencil order (must be even).

    Returns
    -------
    points_idx : list of tuple of int
        List of (ix, iy) index pairs for the four near-corner points:
        lower-left, upper-left, lower-right, upper-right.
    points_coord : list of tuple of float
        List of (x, y) physical coordinates corresponding to each index pair.

    Raises
    ------
    ValueError
        If any index or coordinate is out of bounds or NaN.
    """
    offset = so // 2
    nx, ny = grid.shape

    # Indices for near-corner points
    points_idx = [
        (offset, offset),                # lower-left
        (offset, ny - offset - 1),      # upper-left
        (nx - offset - 1, offset),      # lower-right
        (nx - offset - 1, ny - offset - 1)  # upper-right
    ]

    # Check for NaN in indices
    for idx in points_idx:
        if any(np.isnan(i) for i in idx):
            raise ValueError(f"Index {idx} contains NaN.")

    # Use indices_to_coordinates for robust conversion (raises if out of bounds)
    points_coord = []
    for idx in points_idx:
        coord = indices_to_coordinates(idx, grid)
        if any(np.isnan(c) for c in coord):
            raise ValueError(f"Coordinate {coord} (from index {idx}) contains NaN.")
        points_coord.append(coord)

    return points_idx, points_coord



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
    axes = np.array(axes).reshape(-1)

    for i, (eq, ax) in enumerate(zip(equations, axes)):
        target_func.data[:] = 0
        
        # op_loglevel = 'DEBUG'
        op_loglevel = 'ERROR'
        with devito_log_level(op_loglevel):
            _ = Operator(eq)()

        # setup colormap and normalization
        dmax = max(np.max(target_func.data),len(equations))
        dmin = np.min(target_func.data)
        assert dmin >= 0, "Data contains negative values."
        levels = np.arange(-0.5, dmax + 0.5 + 1, 1)

        cmap = plt.get_cmap(cmap_name, len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)

        # Build a compact symbolic title: full_f = full_f + main_f + left_f + ...
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
        subd_name = getattr(subd, 'name', f'subd{idx}') # subd.name if exists, else 'subd{idx}'
        func = Function(name=f'{subd_name}_f', grid=grid)
        eq = Eq(func, idx, subdomain=subd)   # func = idx in this subdomain (integer value)
        subdomain_functions.append(func)
        subdomain_equations.append(eq)
    
    # Combine all equations: full domain + subdomains
    all_equations = [eq_full] + subdomain_equations

    ###########################################################
    # Run the operator to apply the each equation separately

    # _ = mydebug_op(Operator(all_equations))
    _ = myquiet_op(Operator(all_equations))

    
    ###########################################################
    # For demonstration, create combined equations (step-by-step inclusion of subdomains) 
    import operator
    from itertools import accumulate
    lhs = full_f
    term_list = [eq.lhs for eq in all_equations]
    partial_sums = list(accumulate(term_list[1:], operator.add))
    rhs_list = [lhs + s for s in partial_sums]
    combined_equations = [Eq(lhs, rhs) for rhs in rhs_list]
    
    ###########################################################
    # Run the operator for the combined equations

    # _ = mydebug_op(Operator(combined_equations))
    _ = myquiet_op(Operator(combined_equations))
    
    return combined_equations, full_f