from pathlib import Path
from random import seed

from numpy import dot, array, transpose, corrcoef, exp, zeros, fill_diagonal, searchsorted, ix_, save, load
from numpy.random import permutation
from pandas import read_csv
from scipy.io import loadmat
from scipy.spatial.distance import squareform
from scipy.stats import percentileofscore

from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

seed(2)

data_dir = Path(Path(__file__).parent, "data", "hebart")
n_perms = 10_000


# Compute a p-value by randomisation test
def randomisation_p(rdm_1, rdm_2, observed_r, n_perms):
    r_perms = zeros(n_perms)
    c1 = squareform(rdm_1)
    for perm_i in range(n_perms):
        # if perm_i % 1000 == 0: print(perm_i)
        perm = permutation(48)
        r_perms[perm_i] = corrcoef(
            c1,
            squareform(rdm_2[ix_(perm, perm)])
        )[0, 1]
    p_value = 1 - (percentileofscore(r_perms, observed_r) / 100)
    return p_value


# region Loading and preparing data

# Full embedding and RDM
with Path(data_dir, "spose_embedding_49d_sorted.txt").open() as spose_49_sorted_file:
    embedding_49: array = read_csv(spose_49_sorted_file, skip_blank_lines=True, header=None, delimiter=" ").to_numpy()[:, :49]
dot_product_49 = dot(embedding_49, transpose(embedding_49))
simmat_49 = loadmat(Path(data_dir, 'spose_similarity.mat'))['spose_sim']
rdm_49 = 1 - simmat_49

# 48-object matrices
rdm_48_triplet = loadmat(Path(data_dir, "RDM48_triplet.mat"))["RDM48_triplet"]
rdm_48_triplet_split_half = loadmat(Path(data_dir, "RDM48_triplet_splithalf.mat"))["RDM48_triplet_split2"]

# words and indices
words = [w[0][0] for w in loadmat(Path(data_dir, "words.mat"))["words"]]
words48 = [w[0][0] for w in loadmat(Path(data_dir, "words48.mat"))["words48"]]

# endregion


# region Figure 2
# Indices of the 48 words within the whole list
wordposition48 = searchsorted(words, words48, sorter=array(words).argsort())
cache_spose_sim48 = Path(data_dir, "cache_spose_sim48.npy")
if not cache_spose_sim48.exists():
    # Ported from the Matlab. Just what the hell is this doing? Is this computing pairwise softmax probability?
    esim = exp(dot_product_49)
    cp = zeros((1854, 1854))
    for i in range(1854):
        print(i)
        for j in range(i+1, 1854):
            ctmp = zeros((1, 1854))
            for k_ind in range(len(wordposition48)):
                k = wordposition48[k_ind]
                if (k == i) or (k == j):
                    continue
                ctmp[0, k] = esim[i, j] / ( esim[i, j] + esim[i, k] + esim[j, k] )
            cp[i, j] = ctmp.sum()
    cp /= 48
    cp += transpose(cp)
    fill_diagonal(cp, 1)
    spose_sim48 = cp[ix_(wordposition48, wordposition48)]
    save(cache_spose_sim48, spose_sim48)
else:
    spose_sim48 = load(cache_spose_sim48)

r48 = corrcoef(
    # model dissimilarity matrix
    squareform(1-spose_sim48),
    # "true" dissimilarity matrix
    squareform(rdm_48_triplet))[0, 1]
p_value = randomisation_p(rdm_1=1 - spose_sim48, rdm_2=rdm_48_triplet, observed_r=r48, n_perms=n_perms)
print(f"model vs ppts: {r48}; p={p_value} ({n_perms:,})")  # .89824297

# endregion

# region Generate SM RDM for 48 words

sm = SensorimotorNorms()

# We have the 48 words we need
assert(len(set(sm.iter_words()) & set(words48)))

# try and emulate the mean regularised probability from above

# Get data matrix for all words (that we can)
sm_words = sorted(set(sm.iter_words()) & set(words))
sm_data = sm.matrix_for_words(sm_words)
sm_dotproduct = dot(sm_data, transpose(sm_data))

sm_wordposition48 = searchsorted(sm_words, words48)
cache_sm_sim48 = Path(data_dir, "cache_sm_sim48.npy")
if not cache_sm_sim48.exists():
    esim = exp(sm_dotproduct)
    cp = zeros((len(sm_words), len(sm_words)))
    for i in range(len(sm_words)):
        print(i)
        for j in range(i+1, len(sm_words)):
            ctmp = zeros((1, len(sm_words)))
            for k_ind in range(len(sm_wordposition48)):
                k = sm_wordposition48[k_ind]
                if (k == i) or (k == j):
                    continue
                ctmp[0, k] = esim[i, j] / ( esim[i, j] + esim[i, k] + esim[j, i] )
            cp[i, j] = ctmp.sum()
    cp /= 48
    cp += transpose(cp)
    fill_diagonal(cp, 1)
    sm_sim48 = cp[ix_(sm_wordposition48, sm_wordposition48)]

    save(cache_sm_sim48, sm_sim48)
else:
    sm_sim48 = load(cache_sm_sim48)

sm_r48 = corrcoef(
    squareform(1-sm_sim48),
    squareform(rdm_48_triplet))[0, 1]
p_value = randomisation_p(rdm_1=1 - sm_sim48, rdm_2=rdm_48_triplet, observed_r=sm_r48, n_perms=n_perms)
print(f"sm vs ppts: {sm_r48}; p={p_value} ({n_perms:,})")  # 0.2200492

sm_spose_r48 = corrcoef(
    squareform(1-sm_sim48),
    squareform(1-spose_sim48))[0, 1]
p_value = randomisation_p(rdm_1=1 - sm_sim48, rdm_2=1-spose_sim48, observed_r=sm_spose_r48, n_perms=n_perms)
print(f"sm vs model: {sm_spose_r48}; p={p_value} ({n_perms:,})")  # 0.1899582

# endregion

pass
