"""Utility functions for visualization module."""

import matplotlib.pyplot as plt


def _is_jupyter() -> bool:
    """
    Check if code is running in Jupyter/IPython environment.

    Returns
    -------
    bool
        True if running in Jupyter/IPython, False otherwise.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False

        ipython = get_ipython()
        if 'IPKernelApp' in ipython.config:
            # Running in Jupyter notebook/lab
            return True
        return False
    except (ImportError, AttributeError):
        return False


def _prepare_figure_return(fig: plt.Figure) -> plt.Figure:
    """
    Prepare figure for return to prevent duplicate display in Jupyter.

    In Jupyter notebooks, when a function returns a Figure object,
    it gets automatically displayed. If the figure was already displayed
    during creation (e.g., by seaborn functions), this causes duplication.

    This function closes the figure in Jupyter to prevent auto-display,
    but the closed figure can still be displayed explicitly or saved.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to prepare.

    Returns
    -------
    fig : plt.Figure
        The same figure (possibly closed if in Jupyter).

    Examples
    --------
    >>> def my_plot_function():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3])
    ...     return _prepare_figure_return(fig)
    """
    if _is_jupyter():
        # Close figure to prevent automatic display in Jupyter
        # The figure can still be displayed with display(fig) or saved
        plt.close(fig)

    return fig
