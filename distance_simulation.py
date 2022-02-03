from matplotlib import pyplot
from numpy import histogram, linspace, triu_indices, column_stack
from numpy.random import default_rng, seed
from sklearn.metrics import pairwise_distances

from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from visualisation.distributions import style_histplot

sn = SensorimotorNorms(use_breng_translation=False, verbose=True)

all_ratings = sn.data[sn.VectorColNames].to_numpy()

n_draws = 10_000
seed(451)
rng = default_rng()

matched_dimensions = False

if matched_dimensions:
    fake_vectors = column_stack(tuple(
        rng.choice(all_ratings[:, i], size=n_draws, replace=True)
        for i in range(11)
    ))
else:
    fake_vectors = rng.choice(all_ratings.ravel(), size=(n_draws, 11), replace=True)

fake_distance_matrix = pairwise_distances(fake_vectors, metric="cosine")
fake_distances = fake_distance_matrix[triu_indices(n_draws, k=1)]

bins = linspace(start=0, stop=1,
                    # E.g. 100 bins requires 101 specified bounds
                    num=20 + 1)

h, _ = histogram(fake_distances, bins=bins)

fig, ax = pyplot.subplots(tight_layout=True)
ax.hist(bins[:-1], weights=h, bins=bins)

style_histplot(ax, xlim=(0, 1), ylim=None)

fig.savefig("/Users/caiwingfield/Desktop/dist.png")
pyplot.close(fig)
