from pathlib import Path
from random import seed

from nltk.corpus import wordnet, wordnet_ic
from nltk.corpus.reader import NOUN
from numpy import dot, array, transpose, corrcoef, exp, zeros, fill_diagonal, searchsorted, ix_
from numpy.random import permutation
from pandas import read_csv
from scipy.io import loadmat
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_distances

# from linguistic_distributional_models.utils.logging import print_progress
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

seed(2)

brown_ic = wordnet_ic.ic('ic-brown.dat')

data_dir = Path(Path(__file__).parent, "data", "hebart")
n_perms = 10_000


def mean_softmax_prob_matrix(all_words, select_words, full_similarity_matrix, prefix=""):
    """
    Converts a similarity matrix on a set of conditions to a probability-of-most-similar matrix on a subset of those
    conditions.
    Uses mean softmax probability over triplets of the *selected* conditions containing the given pair.
    Ported from Hebart's Matlab code.
    """
    n_all_conditions = len(all_words)
    n_subset_conditions = len(select_words)
    # Thanks to https://stackoverflow.com/a/33678576/2883198
    sort_idx = array(all_words).argsort()
    word_positions_selected = sort_idx[searchsorted(all_words, select_words, sorter=sort_idx)]
    # Make sure we're not missing any conditions
    assert len(word_positions_selected) == n_subset_conditions

    e_similarity_matrix = exp(full_similarity_matrix)
    cp = zeros((n_all_conditions, n_all_conditions))
    # Hebart et al.'s original code builds the entire matrix for all conditions, then selects out the relevant
    # entries. We can hugely speed up this process by only computing the entries we'll eventually select out.
    for i in word_positions_selected:
        # print_progress(i, n_all_conditions, prefix=prefix)
        for j in word_positions_selected:
            if i == j: continue
            ctmp = zeros((1, n_all_conditions))
            for k in word_positions_selected:
                # Only interested in distinct triplets
                if (k == i) or (k == j):
                    continue
                ctmp[0, k] = (
                        e_similarity_matrix[i, j]
                        / (
                                e_similarity_matrix[i, j]
                                + e_similarity_matrix[i, k]
                                + e_similarity_matrix[j, k]
                        ))
            cp[i, j] = ctmp.sum()
    # print_progress(n_all_conditions, n_all_conditions, prefix=prefix)
    # Complete average
    cp /= n_subset_conditions
    # Fill in the rest of the symmetric similarity matrix
    # cp += transpose(cp)  # No longer need to do this now we're filling in both sides of the matrix in the above loop
    # Instead we fix rounding errors by forcing symmetry
    cp += transpose(cp); cp /= 2
    fill_diagonal(cp, 1)
    # Select out words of interest
    selected_similarities = cp[ix_(word_positions_selected, word_positions_selected)]
    return selected_similarities


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


def similarity(word_1, word_2):
    synsets1 = wordnet.synsets(word_1, pos=NOUN)
    synsets2 = wordnet.synsets(word_2, pos=NOUN)
    max_jcn_distance = 0
    for s1 in synsets1:
        for s2 in synsets2:
            jcn = s1.jcn_similarity(s2, brown_ic)
            max_jcn_distance = max(max_jcn_distance, jcn)
    return max_jcn_distance


def compute_wordnet_sm(words):
    n_words = len(words)
    similarity_matrix = zeros((n_words, n_words))
    for i in range(n_words):
        for j in range(n_words):
            similarity_matrix[i, j] = similarity(words[i], words[j])
    fill_diagonal(similarity_matrix, 1)
    return similarity_matrix


def main():
    # region Loading and preparing data

    # words and indices
    words = [w[0][0] for w in loadmat(Path(data_dir, "words.mat"))["words"]]
    words48 = [w[0][0] for w in loadmat(Path(data_dir, "words48.mat"))["words48"]]

    # Full embedding and RDM
    with Path(data_dir, "spose_embedding_49d_sorted.txt").open() as spose_49_sorted_file:
        embedding_49: array = read_csv(spose_49_sorted_file, skip_blank_lines=True, header=None, delimiter=" ").to_numpy()[:, :49]

    # Full similarity matrices
    dot_product_49 = dot(embedding_49, transpose(embedding_49))
    dot_product_49_11 = dot(embedding_49[:, :11], transpose(embedding_49[:, :11]))
    dot_product_49_bottom_11 = dot(embedding_49[:, -11:], transpose(embedding_49[:, -11:]))

    # 48-object participant RDM
    rdm48_participant = loadmat(Path(data_dir, "RDM48_triplet.mat"))["RDM48_triplet"]

    # endregion

    # region Figure 2

    # Indices of the 48 words within the whole list
    spose_sim48 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49, prefix="SPOSE")
    rdm48_spose = 1 - spose_sim48

    r48 = corrcoef(
        # model dissimilarity matrix
        squareform(rdm48_spose),
        # "true" dissimilarity matrix
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose, rdm_2=rdm48_participant, observed_r=r48, n_perms=n_perms)
    print(f"model vs ppts: {r48}; p={p_value} ({n_perms:,})")  # .89824297

    spose_sim48_11 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49_11, prefix="SPOSE 11")
    rdm48_spose_11 = 1 - spose_sim48_11

    r48_11 = corrcoef(
        squareform(rdm48_spose_11),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose_11, rdm_2=rdm48_participant, observed_r=r48_11, n_perms=n_perms)
    print(f"model[11] vs ppts: {r48_11}; p={p_value} ({n_perms:,})")

    spose_sim48_bottom_11 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49_bottom_11, prefix="SPOSE bottom_11")
    rdm48_spose_bottom_11 = 1 - spose_sim48_bottom_11

    r48_bottom_11 = corrcoef(
        squareform(rdm48_spose_bottom_11),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose_bottom_11, rdm_2=rdm48_participant, observed_r=r48_bottom_11, n_perms=n_perms)
    print(f"model[bottom_11] vs ppts: {r48_bottom_11}; p={p_value} ({n_perms:,})")

    # endregion

    # region Generate SM RDM for 48 words

    sm = SensorimotorNorms()

    # try and emulate the mean regularised probability from above

    # Get data matrix for all words (that we can)
    sm_data = sm.matrix_for_words(words48)

    sm_rdm_cosine = cosine_distances(sm_data, sm_data)
    sm_sim48_cosine = mean_softmax_prob_matrix(all_words=words48, select_words=words48, full_similarity_matrix=1 - sm_rdm_cosine, prefix="SM cosine")
    rdm48_sensorimotor_cosine = 1-sm_sim48_cosine

    sm_r48_cosine = corrcoef(
        squareform(rdm48_sensorimotor_cosine),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_sensorimotor_cosine, rdm_2=rdm48_participant, observed_r=sm_r48_cosine, n_perms=n_perms)
    print(f"sm_cosine vs ppts: {sm_r48_cosine}; p={p_value} ({n_perms:,})")  # .3163023620890325
    print(f"  Without softmax: {corrcoef(squareform(sm_rdm_cosine), squareform(rdm48_participant))[0,1]}")

    sm_rdm_minkowski = distance_matrix(sm_data, sm_data, p=3)
    sm_sm_minkowski = 1 - (sm_rdm_minkowski / max(sm_rdm_minkowski.flatten()[:])); fill_diagonal(sm_sm_minkowski, 1)
    sm_sim48_minkowski = mean_softmax_prob_matrix(all_words=words48, select_words=words48, full_similarity_matrix=sm_sm_minkowski, prefix="SM Minkowski-3")
    rdm48_sensorimotor_minkowski = 1 - sm_sim48_minkowski

    sm_r48_minkowski = corrcoef(
        squareform(rdm48_sensorimotor_minkowski),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_sensorimotor_minkowski, rdm_2=rdm48_participant, observed_r=sm_r48_minkowski, n_perms=n_perms)
    print(f"sm_minkowski vs ppts: {sm_r48_minkowski}; p={p_value} ({n_perms:,})")  # .2704680432064961
    print(f"  Without softmax: {corrcoef(squareform(sm_rdm_minkowski), squareform(rdm48_participant))[0,1]}")

    sm_sm_dotpropduct = dot(sm_data, transpose(sm_data))
    sm_sim48_dotproduct = mean_softmax_prob_matrix(all_words=words48, select_words=words48, full_similarity_matrix=sm_sm_dotpropduct, prefix="SM dotproduct")
    rdm48_sensorimotor_dotproduct = 1 - sm_sim48_dotproduct

    sm_r48_dotproduct = corrcoef(
        squareform(rdm48_sensorimotor_dotproduct),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_sensorimotor_dotproduct, rdm_2=rdm48_participant, observed_r=sm_r48_dotproduct, n_perms=n_perms)
    print(f"sm_dotproduct vs ppts: {sm_r48_dotproduct}; p={p_value} ({n_perms:,})")  # .2821415691953627

    wordnet_sm = compute_wordnet_sm(words=words48)
    wordnet_sim48 = mean_softmax_prob_matrix(all_words=words48, select_words=words48, full_similarity_matrix=wordnet_sm, prefix="Wordnet")
    rdm48_wordnet = 1 - wordnet_sim48

    wordnet_r48 = corrcoef(
        squareform(rdm48_wordnet),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_wordnet, rdm_2=rdm48_participant, observed_r=wordnet_r48, n_perms=n_perms)
    print(f"wordnet vs ppts: {wordnet_r48}; p={p_value} ({n_perms:,})")
    print(f"  Without softmax: {corrcoef(squareform(1-wordnet_sm), squareform(rdm48_participant))[0,1]}")

    # endregion

    pass


if __name__ == '__main__':
    main()
