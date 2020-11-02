import matplotlib.pyplot as plt
import numpy as np


def generate_hist(data, range=None, bins=20, density=None):
    """ Compute the histogram of a set of data.

    Args:
        data (np.ndarray): One dim data.
        bins (int, sequence, optional): If `bins` is an int, it defines the number
            of equal-width bins in the given range (10, by default). If `bins`
            is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.
        density (bool, optional): If ``False``, the result will contain the number of samples in
            each bin. If ``True``, the result is the value of the probability *density*
            function at the bin, normalized such that the *integral* over the range is 1.
        range (tuple, optional): The lower and upper range of the bins.  If not provided, range
            is simply ``(a.min(), a.max())``.  Values outside the range are ignored.
    """

    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if range is None:
        range = [np.min(data), np.max(data)]
        if not isinstance(bins, int):
            range[0] = min(np.min(bins), min(range))
            range[1] = max(np.max(bins), max(range))
    else:
        range = range
    percentage = len(data[(data >= range[0]) & (data <= range[1])]) / len(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=density, range=range)
    return percentage, hist, bin_edges, range


def cdf(data,
        bins=20,
        density=None,
        label_scale=1,
        rotation=30,
        range=None,
        color=None):
    """ A function to plot cumulative probability density.

    Args:
        data (np.ndarray): One dim data.
        bins (int, sequence, optional): If `bins` is an int, it defines the number
            of equal-width bins in the given range (10, by default). If `bins`
            is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.
        density (bool, optional): If ``False``, the result will contain the number of samples in
            each bin. If ``True``, the result is the value of the probability *density*
            function at the bin, normalized such that the *integral* over the range is 1.
        label_scale (float, optional): Scale of label in axis.
        rotation (float, int, optional): Tick label rotation.
        range (tuple, optional): The lower and upper range of the bins.  If not provided, range
            is simply ``(a.min(), a.max())``.  Values outside the range are ignored.
        color (str, optional): Color of line.
    """
    percent, hist, bin_edges, range = generate_hist(data, range, bins, density)

    cum_sum_hist = np.cumsum(hist / sum(hist) * percent)
    plt.xticks(bin_edges[1:])
    plt.tick_params(axis='x', labelsize=5 * label_scale, labelrotation=rotation)
    width = -(bin_edges[1] - bin_edges[0])
    if color is None:
        plt.plot(bin_edges[1:], cum_sum_hist, '-*')
    else:
        plt.plot(bin_edges[1:], cum_sum_hist, '-*', color=color)
    x_shift = width * 0.5
    y_shift = 0.02
    for i, (a, b) in enumerate(zip(bin_edges[1:], cum_sum_hist)):
        if a < range[0] or a > range[1]:
            continue
        plt.text(a + x_shift, b + y_shift, '%.3f' % b, color='#ED7D31',
                 ha='center', va='bottom', fontsize=10 * label_scale, rotation=rotation)
    plt.xlim(range)


def pdf(data,
        bins=20,
        density=None,
        label_scale=1,
        rotation=30,
        range=None,
        color=None):
    """ A function to plot cumulative probability density.

    Args:
        data (np.ndarray): One dim data.
        bins (int, sequence, optional): If `bins` is an int, it defines the number
            of equal-width bins in the given range (10, by default). If `bins`
            is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.
        density (bool, optional): If ``False``, the result will contain the number of samples in
            each bin. If ``True``, the result is the value of the probability *density*
            function at the bin, normalized such that the *integral* over the range is 1.
        label_scale (float, optional): Scale of label in axis.
        rotation (float, int, optional): Tick label rotation.
        range (tuple, optional): The lower and upper range of the bins.  If not provided, range
            is simply ``(a.min(), a.max())``.  Values outside the range are ignored.
        color (str, optional): Color of line.
    """

    percent, hist, bin_edges, range = generate_hist(data, range, bins, density)
    plt.xticks(bin_edges[1:])
    plt.tick_params(axis='x', labelsize=5 * label_scale, labelrotation=rotation)
    width = -(bin_edges[1] - bin_edges[0])
    if color is None:
        plt.bar(bin_edges[1:], hist / sum(hist),
                width=width,
                align='edge')
    else:
        plt.bar(bin_edges[1:], hist / sum(hist),
                width=width,
                align='edge',
                color=color)

    x_shift = width * 0.5
    y_shift = 0.02
    for i, (a, b) in enumerate(zip(bin_edges[1:], hist / sum(hist))):
        if a < range[0] or a > range[1]:
            continue
        plt.text(a + x_shift, b + y_shift, '%.3f' % b,
                 ha='center', va='bottom', fontsize=10 * label_scale, rotation=rotation)
    plt.xlim(range)


def cdf_pdf(data,
            bins=20,
            density=None,
            label_scale=1,
            rotation=30,
            percentile=False,
            percent=0.02,
            range=None,
            c_color=None,
            p_color=None,
            ):
    """ A function to plot cumulative probability density.

    Args:
        data (np.ndarray): One dim data.
        bins (int, sequence, optional): If `bins` is an int, it defines the number
            of equal-width bins in the given range (10, by default). If `bins`
            is a sequence, it defines a monotonically increasing array of bin edges,
            including the rightmost edge, allowing for non-uniform bin widths.
        density (bool, optional): If ``False``, the result will contain the number of samples in
            each bin. If ``True``, the result is the value of the probability *density*
            function at the bin, normalized such that the *integral* over the range is 1.
        label_scale (float, optional): Scale of label in axis.
        rotation (float, int, optional): Tick label rotation.
        percentile (str, optional): Using a line to show where is the percentage data.
        percent (float, optional) : Percent data.
        range (tuple, optional): The lower and upper range of the bins.  If not provided, range
            is simply ``(a.min(), a.max())``.  Values outside the range are ignored.
        c_color (str, optional): Color of cumsum line.
        b_color (str, optional): Color of histograms.
    """

    cdf(data, bins, density, label_scale, rotation, range, c_color)
    pdf(data, bins, density, label_scale, rotation, range, p_color)

    if percentile:
        if not percentile in ['min', 'max', 'both']:
            raise ValueError('percentile must be in "min", "max" or "both"')
        min_value = np.percentile(data, percent * 100)
        max_value = np.percentile(data, (1 - percent) * 100)
        if percentile == 'min':
            plt.axvline(x=min_value, ls='-.', color='#66CCFF')
            plt.text(min_value, 0.8, f'{percent * 100}%\npercentile:\n{min_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
        if percentile == 'max':
            plt.axvline(x=max_value, ls='-.', color='#66CCFF')
            plt.text(max_value, 0.8, f'{(1 - percent) * 100}%\npercentile:\n{max_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
        if percentile == 'both':
            plt.axvline(x=min_value, ls='-.', color='#66CCFF')
            plt.text(min_value, 0.8, f'{percent * 100}%\npercentile:\n{min_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
            plt.axvline(x=max_value, ls='-.', color='#66CCFF')
            plt.text(max_value, 0.8, f'{(1 - percent) * 100}%\npercentile:\n{max_value:.2f}',
                     ha='center', va='bottom', fontsize=12 * label_scale)
