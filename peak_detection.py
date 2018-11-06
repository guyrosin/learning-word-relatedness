import numpy as np
from scipy.signal import argrelextrema

min_peak_value = 0.3
plateau_threshold = 0.15
relative_threshold = 0.9


def pad_with_zeros(hist_list):
    """ For each year which doesn't exist here, put 0 """
    last_year = hist_list[0][0] - 1  # initialize to be less than the first year
    i = 0
    while i < len(hist_list):
        year_item = hist_list[i]
        if year_item[0] - last_year > 1:
            # fill the gap
            while year_item[0] - last_year > 1:
                last_year += 1
                hist_list.insert(i, (last_year, 0))
                i += 1
        last_year += 1
        i += 1
    return hist_list


def find_peaks(hist):
    """ Gets a dictionary of tuples: (year, value). returns a list of peak years """
    if not hist:
        return []
    # sort the histogram and convert to list (for the graph)
    hist_list = list(sorted(hist.items(), key=lambda t: t[0]))
    pad_with_zeros(hist_list)
    hist_list = np.array([[year, value] for (year, value) in hist_list])
    values = hist_list[:, 1]
    peak_indices = argrelextrema(values, np.greater_equal)[0]
    peaks = hist_list[peak_indices]
    peaks = np.array([[year, value] for [year, value] in peaks if value > min_peak_value])
    if peaks.size == 0:
        return np.array([])
    final_peak_indices = np.array([], dtype=int)
    max_peak_value = np.amax(peaks[:, 1])
    for i in peak_indices:
        value = hist_list[i, 1]
        # filter peaks that are much lower than the highest peak
        if value < relative_threshold * max_peak_value:
            continue
        final_peak_indices = np.append(final_peak_indices, i)
        is_plateau = True
        # look to the right
        j = i + 1
        while is_plateau and j < len(hist_list):
            if hist_list[j, 1] > min_peak_value and abs(value / hist_list[j, 1]) - 1 < plateau_threshold:
                final_peak_indices = np.append(final_peak_indices, j)
                j += 1
            else:
                is_plateau = False
        # look to the left
        is_plateau = True
        j = i - 1
        while is_plateau and j >= 0:
            if hist_list[j, 1] > min_peak_value and abs(value / hist_list[j, 1]) - 1 < plateau_threshold:
                final_peak_indices = np.append(final_peak_indices, j)
                j -= 1
            else:
                is_plateau = False

    final_peak_indices = np.sort(np.unique(final_peak_indices))
    peak_years = hist_list[final_peak_indices][:, 0]
    return peak_years
