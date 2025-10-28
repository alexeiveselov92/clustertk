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
    Return figure object for display.

    This function simply returns the figure object, allowing matplotlib's
    standard display behavior in both Jupyter and regular Python environments.

    In Jupyter: figure auto-displays if it's the last expression in a cell.
    In scripts: use plt.show() or save the figure.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to return.

    Returns
    -------
    fig : plt.Figure
        The same figure object.

    Examples
    --------
    >>> def my_plot_function():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3])
    ...     return _prepare_figure_return(fig)
    """
    return fig
