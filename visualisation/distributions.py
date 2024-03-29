from pathlib import Path
from typing import Tuple, Optional, Dict

import yaml
from matplotlib.axes import Axes
from numpy import array, zeros, arange, repeat, linspace, savetxt, loadtxt, histogram, inf, nditer, sqrt
from numpy.linalg import inv
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix
from matplotlib import pyplot

from predictors.aux import logger
from predictors.distance import Distance, Minkowski3, Mahalanobis, Cosine, Correlation
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

sn = SensorimotorNorms()


def bin_distances(bins, distance: Distance) -> Tuple[array, float, float, float, float]:
    """
    Get all pairwise distances from the model and graph a histogram distribution using the specified bins.

    :param bins: array of bin inclusive lower-bounds plus the inclusive upper bound of the last bin (e.g. from linspace)
    :param distance:
    :return: tuple:
        binned distances (array of counts of distances for each bin),
        min attained distance
        max attained distance
        mean distance
        sd distance
    """

    n_bins = len(bins) - 1
    all_words = list(sn.iter_words())

    # We don't want to get the entire 40k-by-40k distance matrix into memory at once, so instead we can pre-define the
    # bins for the histograms, and accumulate the totals a bit at a time.

    min_attained_distance = inf
    max_attained_distance = 0
    binned_distances = zeros(n_bins,
                             # floats to avoid broadcasting errors on divides below.
                             # We'll cast to int later
                             dtype=float)
    count = 0
    cumulative_mean = 0  # will store the cumulatively computed mean
    cumulative_ssd = 0  # will store the cumulatively computed sum of squared differences from the mean
    for i, word in enumerate(all_words):
        if i % 1_000 == 0:
            logger.info(f"Done {i:,} words")

        word_vector = array(sn.sensorimotor_vector_for_word(word))
        all_data = array(sn.matrix_for_words(all_words))

        if isinstance(distance, Minkowski3):
            distances_this_word: array = minkowski_distance_matrix(word_vector.reshape(1, 11), all_data, 3).flatten()
        elif isinstance(distance, Mahalanobis):
            distances_this_word: array = distance_matrix(word_vector.reshape(1, 11), all_data, metric=distance.name, VI=inv(distance.covariance_matrix)).flatten()
        else:
            distances_this_word: array = distance_matrix(word_vector.reshape(1, 11), all_data, metric=distance.name).flatten()

        binned_distances_this_word, _ = histogram(distances_this_word, bins)
        binned_distances += binned_distances_this_word

        min_attained_distance = min(min_attained_distance,
                                    # for minimum distances, we don't want to include the 0 from the identity comparison
                                    # thanks to https://stackoverflow.com/a/19286855/2883198
                                    distances_this_word[arange(len(distances_this_word)) != i].min())
        max_attained_distance = max(max_attained_distance, distances_this_word.max())

        # Mean and SD for streaming data
        # Thanks to https://nestedsoftware.com/2018/03/20/calculating-a-moving-average-on-streaming-data-5a7k.22879.html
        for d in nditer(distances_this_word):
            count += 1
            mean_differential = (d - cumulative_mean) / count
            new_mean = cumulative_mean + mean_differential
            d2_increment = (d - new_mean) * (d - cumulative_mean)
            cumulative_mean = new_mean
            cumulative_ssd += d2_increment

    # We've double-counted many of the distances by matching word X with all words (including Y) and word Y with all
    # words (including X). However we haven't double-counted the diagonal (each word X was matched with all words,
    # including X, but only once).
    # Therefore we need to do a bit of arithmetic to adjust the totals appropriately.
    # For every bin except for the one containing 0, we need to halve the totals.
    binned_distances[1:] /= 2
    # For the first bin, which includes all the 0 distances of identity pairs, we need to be cleverer
    n_identity_pairs = len(all_words)
    binned_distances[0] = (
        # Half of the counts for the non-identity pairs
        ((binned_distances[0] - n_identity_pairs) / 2)
        # And all of the identity pairs
        + n_identity_pairs
    )
    assert (binned_distances == binned_distances.astype(int)).all()

    # Similarity for the mean, we've double-counted lots, but not including the diagonal zeros.
    # So we need to adjust the mean accordingly:
    n_ltv_pairs = len(all_words) * (len(all_words) + 1) / 2  # including the diagonal
    mean = (cumulative_mean
            # `count` is the total number of elements in the square distance matrix, so multiplying gives us the sum.
            * count
            # We only want to single-count entries for the sum, and the diagonal entries didn't contribute, so we can
            # just divide by 2.
            / 2
            # Finally we divide by the number of single-counted ltv entries, INCLUDING the diagonal (the +1).
            / n_ltv_pairs)

    # And for the SD, we also have double-counted just the non-zero entries.
    # cumulative_ssd holds the sum of squared distances from the mean. we can just halve this to get the ssd of the
    # single-counted entries, and complete the rest of the sample sd formula as required.
    sd = sqrt(cumulative_ssd / 2 / (n_ltv_pairs - 1))

    return binned_distances.astype(int), min_attained_distance, max_attained_distance, mean, sd


def style_histplot(ax: Axes, xlim: Tuple[float, float], ylim: Optional[Tuple[float, float]]):
    # noinspection PyTypeChecker
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def graph_sensorimotor_distance_distribution(distance: Distance, n_bins: int, location: Path, overwrite: bool,
                                             ylim: Optional[Tuple[float, float]]):
    """
    Graph the distribution of sensorimotor distances among all pairs of concepts in the norms.

    :param distance:
    :param n_bins:
    :param ylim:
    :param location:
    :param overwrite:
    :return:
    """

    figure_save_path = Path(location, f"distance distribution {distance.name} {n_bins} bins.svg")
    distribution_save_path = Path(location, f"distance distribution {distance.name} {n_bins} bins.txt")
    descriptive_stats_path = Path(location, f"distance {distance.name} descriptive.yaml")

    min_distance = 0
    if isinstance(distance, Cosine):
        # It's 1 not 2 because all values are positive so the furthest apart we can get is tau/4
        max_distance = 1.0
    elif isinstance(distance, Correlation):
        # We can in theory get pairs of anticorrelated vectors, e.g. linspace(0, 5, 11) and linspace(5, 0, 11)
        max_distance = 2.0
    else:
        max_distance = distance.distance(repeat(sn.rating_min, 11), repeat(sn.rating_max, 11))

    # [:-1] determine the inclusive lower-bounds of each bin.
    # [-1] determines the inclusive upper bound of the last bin
    bins = linspace(start=min_distance, stop=max_distance,
                    # That's why we need the +1 here
                    # E.g. 100 bins requires 101 specified bounds
                    num=n_bins + 1)

    if distribution_save_path.exists() and not overwrite:
        logger.warning(f"{distribution_save_path} exists, skipping")
        with distribution_save_path.open("r") as distribution_file:
            binned_distances = loadtxt(distribution_file)
        with descriptive_stats_path.open("r") as descriptive_file:
            descriptive_stats: Dict[str, float] = yaml.load(descriptive_file, yaml.SafeLoader)
        min_attained_distance = descriptive_stats["Minimum attained distance"]
        max_attained_distance = descriptive_stats["Maximum attained distance"]
        mean_distance         = descriptive_stats["Mean distance"]
        sd_distance           = descriptive_stats["SD distance"]
    else:
        binned_distances, min_attained_distance, max_attained_distance, mean_distance, sd_distance = bin_distances(bins, distance)
        with distribution_save_path.open("w") as distribution_file:
            savetxt(distribution_file, binned_distances)
        with descriptive_stats_path.open("w") as descriptive_file:
            yaml.dump({
                "Minimum attained distance": float(min_attained_distance),
                "Maximum attained distance": float(max_attained_distance),
                "Mean distance":             float(mean_distance),
                "SD distance":               float(sd_distance),
            }, descriptive_file, yaml.SafeDumper)

    logger.info(f"Max theoretical pairwise {distance.name} distance between concepts: {max_distance}")
    logger.info(f"Attained {distance.name} distance range: [{min_attained_distance}, {max_attained_distance}]")
    logger.info(f"Mean (SD) {distance.name} distance: {mean_distance} ({sd_distance})")

    fig, ax = pyplot.subplots(tight_layout=True)
    ax.hist(
        # Everything gets set to the left-hand edge
        bins[:-1],
        weights=binned_distances, bins=bins)

    style_histplot(ax, xlim=(min_distance, max_distance), ylim=ylim)

    fig.savefig(figure_save_path)
    pyplot.close(fig)
