from pathlib import Path
from typing import Tuple

from matplotlib.axes import Axes
from numpy import repeat, linspace, loadtxt, zeros, histogram, array, savetxt
from scipy.spatial import distance_matrix as minkowski_distance_matrix
from scipy.spatial.distance import cdist as distance_matrix
from matplotlib import pyplot

from linguistic_distributional_models.utils.maths import DistanceType, distance
from predictors.aux import logger
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

sn = SensorimotorNorms()


def bin_distances(bins, distance_type, n_bins):
    """
    We don't want to get the entire 40k-by-40k distance matrix into memory at once, so instead we can pre-define the
    bins for the histograms, and accumulate the totals a bit at a time.

    :param bins:
    :param distance_type:
    :param n_bins:
    :return:
    """
    binned_distances = zeros(n_bins - 1, dtype=int)
    all_words = list(sn.iter_words())
    for i, word in enumerate(all_words):
        if i % 1_000 == 0:
            logger.info(f"Done {i:,} words")

        word_vector = array(sn.vector_for_word(word))
        all_data = array(sn.matrix_for_words(all_words))

        if distance_type == DistanceType.Minkowski3:
            distances: array = minkowski_distance_matrix(word_vector.reshape(1, 11), all_data, 3)
        else:
            distances: array = distance_matrix(word_vector.reshape(1, 11), all_data, metric=distance_type.name)

        htemp, _ = histogram(distances, bins)
        binned_distances += htemp
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
    return binned_distances


def style_histplot(ax: Axes, xlim: Tuple[float, float]):
    # noinspection PyTypeChecker
    ax.set_xlim(xlim)


def graph_distance_distribution(distance_type: DistanceType, n_bins: int, location: Path, overwrite: bool):
    """
    Graph the distribution of distances among all pairs of concepts in the norms.

    :param distance_type:
    :param n_bins:
    :param location:
    :param overwrite:
    :return:
    """

    figure_save_path = Path(location, f"distance distribution {distance_type.name} {n_bins} bins.svg")
    distribution_save_path = Path(location, f"distance distribution {distance_type.name} {n_bins} bins.csv")

    min_distance = 0
    if distance_type == distance_type.cosine:
        # It's 1 not 2 because all values are positive so the furthest apart we can get is tau/4
        max_distance = 1.0
    elif distance_type == distance_type.correlation:
        # We can in theory get pairs of anticorrelated vectors, e.g. linspace(0, 5, 11) and linspace(5, 0, 11)
        max_distance = 2.0
    else:
        max_distance = distance(repeat(sn.rating_min, 11), repeat(sn.rating_max, 11), distance_type=distance_type)
    logger.info(f"Max theoretical pairwise {distance_type.name} distance between concepts: {max_distance}")

    bins = linspace(start=min_distance, stop=max_distance, num=n_bins)

    if distribution_save_path.exists() and not overwrite:
        logger.warning(f"{distribution_save_path} exists, skipping")
        with distribution_save_path.open("r") as distribution_file:
            binned_distances = loadtxt(distribution_file)
    else:
        binned_distances = bin_distances(bins, distance_type, n_bins)
        with distribution_save_path.open("w") as distribution_file:
            savetxt(distribution_file, binned_distances)

    fig, ax = pyplot.subplots(tight_layout=True)
    ax.hist(bins[:-1], weights=binned_distances, bins=bins)

    style_histplot(ax, xlim=(min_distance, max_distance))

    fig.savefig(figure_save_path)
    pyplot.close(fig)


def graph_distance_distributions(n_bins: int, location: Path, overwrite: bool):
    graph_distance_distribution(distance_type=DistanceType.cosine, n_bins=n_bins, location=location, overwrite=overwrite)
    graph_distance_distribution(distance_type=DistanceType.correlation, n_bins=n_bins, location=location, overwrite=overwrite)
    graph_distance_distribution(distance_type=DistanceType.Minkowski3, n_bins=n_bins, location=location, overwrite=overwrite)
    graph_distance_distribution(distance_type=DistanceType.Euclidean, n_bins=n_bins, location=location, overwrite=overwrite)
