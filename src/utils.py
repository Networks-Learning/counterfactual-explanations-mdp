import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_fig_dim(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / golden_ratio

    fig_dim = (fig_height_in, golden_ratio)

    return fig_dim


def latexify(font_size=10, legend_font_size=9):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    params = {'backend': 'ps',
              'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
              # fontsize for x and y labels (was 10)
            #   'axes.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'axes.titlesize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'font.size': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            #   'legend.fontsize': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            #   'xtick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            #   'ytick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'font.size': font_size,  # was 10
              'legend.fontsize': legend_font_size,  # was 10
              'legend.title_fontsize': legend_font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'text.usetex': True,
            #   'figure.figsize': [fig_width, fig_height],
              'font.family' : 'serif',
              'font.serif' : 'Computer Modern',
              'mathtext.fontset' : 'cm'
            #   'xtick.minor.size': 0.5,
            #   'xtick.major.pad': 1.5,
            #   'xtick.major.size': 1,
            #   'ytick.minor.size': 0.5,
            #   'ytick.major.pad': 1.5,
            #   'ytick.major.size': 1,
            # #   'lines.linewidth': 1.5,
            # 'lines.linewidth': 1,
            # #   'lines.markersize': 0.1,
            #   'lines.markersize': 8.0,
            #   'hatch.linewidth': 0.5
              }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
