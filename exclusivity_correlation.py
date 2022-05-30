"""
===========================
Computes the correlation between pairwise distances and mean exclusivity ratings for randomly drawn pairs of norms.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2022
---------------------------
"""

from numpy import corrcoef, zeros
from numpy.random import default_rng, seed

from linguistic_distributional_models.utils.logging import print_progress
from predictors.distance import Distance, Cosine
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms, DataColNames

sn = SensorimotorNorms(use_breng_translation=False, verbose=True)


def exclusivity_correlation(n_draws: int, distance: Distance):
    rng = default_rng()

    all_words = list(sn.iter_words())
    random_words = rng.choice(all_words, 2 * n_draws, replace=True)
    first_words = random_words[:n_draws]
    second_words = random_words[n_draws:]

    distances = zeros((n_draws,))  # Preallocate vectors to be correlated
    mean_exclusivities = zeros((n_draws,))
    for i in range(n_draws):
        w1, w2 = first_words[i], second_words[i]
        v1, v2 = sn.sensorimotor_vector_for_word(w1), sn.sensorimotor_vector_for_word(w2)
        e1, e2 = sn.stat_for_word(w1, DataColNames.exclusivity_sensorimotor), sn.stat_for_word(w2, DataColNames.exclusivity_sensorimotor)

        # For the pair
        distances[i] = distance.distance(v1, v2)  # vector distance
        mean_exclusivities[i] = (e1 + e2) / 2  # mean exclusivity

        print_progress(i + 1, n_draws)

    return corrcoef(distances, mean_exclusivities)


if __name__ == "__main__":
    seed(451)
    correlation = exclusivity_correlation(n_draws=10_000, distance=Cosine())
    print(correlation)
