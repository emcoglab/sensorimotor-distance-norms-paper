from numpy import corrcoef, zeros
from numpy.random import default_rng, seed

from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType, distance
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms, DataColNames

sn = SensorimotorNorms(use_breng_translation=False, verbose=True)

n_draws = 10_000
seed(451)
rng = default_rng()

all_words = list(sn.iter_words())

random_words = rng.choice(all_words, 2 * n_draws, replace=True)
w1s = random_words[:n_draws]
w2s = random_words[n_draws:]

ds = zeros((n_draws, ))
es = zeros((n_draws, ))
for i in range(n_draws):
    w1, w2 = w1s[i], w2s[i]
    v1, v2 = sn.sensorimotor_vector_for_word(w1), sn.sensorimotor_vector_for_word(w2)
    e1, e2 = sn.stat_for_word(w1, DataColNames.exclusivity_sensorimotor), sn.stat_for_word(w2, DataColNames.exclusivity_sensorimotor)

    # For the pair
    ds[i] = distance(v1, v2, DistanceType.cosine)  # vector distance
    es[i] = (e1 + e2) / 2  # mean exclusivity

    print_progress(i+1, n_draws)

print(corrcoef(ds, es))
