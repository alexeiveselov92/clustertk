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
    Prepare figure for return to prevent duplicate display.

    When using plt.subplots(), the figure is registered in pyplot's global state.
    In Jupyter, this causes the figure to auto-display. If we then return the figure,
    Jupyter displays it again, resulting in duplication.

    Solution: plt.close(fig) removes the figure from pyplot's state, but the figure
    object remains fully functional - it can still be displayed, saved, or manipulated.

    This matches the behavior expected by users and prevents duplication.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to prepare.

    Returns
    -------
    fig : plt.Figure
        The figure object (removed from pyplot state to prevent duplication).

    Examples
    --------
    >>> def my_plot_function():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3])
    ...     return _prepare_figure_return(fig)
    ...
    >>> # In Jupyter: displays once
    >>> fig = my_plot_function()
    >>> # Can still save or manipulate
    >>> fig.savefig('plot.png')
    """
    # Remove figure from pyplot state to prevent duplicate display
    # The figure object remains usable (can be displayed, saved, etc.)
    plt.close(fig)
    return fig
