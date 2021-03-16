from pathlib import Path

from numpy import dot, array, transpose, corrcoef, fill_diagonal, ix_
from pandas import read_csv
from scipy.io import loadmat
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances

from predictors.aux import find_indices, hebart_dir
from predictors.rsa import randomisation_p, mean_softmax_prob_matrix, compute_wordnet_sm, compute_lsa_sm, \
    compute_buchanan_sm
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

n_perms = 10_000


def main():
    # region Loading and preparing data

    # words and indices
    words = [w[0][0] for w in loadmat(Path(hebart_dir, "words.mat"))["words"]]
    words48 = [w[0][0] for w in loadmat(Path(hebart_dir, "words48.mat"))["words48"]]

    # Full embedding and RDM
    with Path(hebart_dir, "spose_embedding_49d_sorted.txt").open() as spose_49_sorted_file:
        embedding_49: array = read_csv(spose_49_sorted_file, skip_blank_lines=True, header=None, delimiter=" ").to_numpy()[:, :49]

    # Full similarity matrices
    dot_product_49 = dot(embedding_49, transpose(embedding_49))
    dot_product_49_11 = dot(embedding_49[:, :11], transpose(embedding_49[:, :11]))
    dot_product_49_bottom_11 = dot(embedding_49[:, -11:], transpose(embedding_49[:, -11:]))

    # 48-object participant RDM
    rdm48_participant = loadmat(Path(hebart_dir, "RDM48_triplet.mat"))["RDM48_triplet"]

    # endregion

    # region Figure 2

    subset = True

    buchanan_sm, words18_buchanan = compute_buchanan_sm(words=words48)
    print(words18_buchanan)
    words18_idxs = find_indices(words48, words18_buchanan)
    rdm18_participant = rdm48_participant[ix_(words18_idxs, words18_idxs)]
    buchanan_sim18 = mean_softmax_prob_matrix(all_words=words18_buchanan, full_similarity_matrix=buchanan_sm)
    rdm18_buchanan = 1 - buchanan_sim18

    buchanan_r18 = corrcoef(
        squareform(rdm18_buchanan),
        squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_buchanan, rdm_2=rdm18_participant, observed_r=buchanan_r18, n_perms=n_perms)
    print(f"buchanan vs ppts: {buchanan_r18}; p={p_value} ({n_perms:,})")
    print(f"  Without softmax: {corrcoef(squareform(1-buchanan_sm), squareform(rdm18_participant))[0,1]}")

    # Indices of the 48 words within the whole list
    spose_sim48 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49)
    rdm48_spose = 1 - spose_sim48
    spose_sim18 = mean_softmax_prob_matrix(all_words=words, select_words=words18_buchanan, full_similarity_matrix=dot_product_49)
    rdm18_spose = 1 - spose_sim18

    r48_spose = corrcoef(
        # model dissimilarity matrix
        squareform(rdm48_spose),
        # "true" dissimilarity matrix
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose, rdm_2=rdm48_participant, observed_r=r48_spose, n_perms=n_perms)
    print(f"model vs ppts: {r48_spose}; p={p_value} ({n_perms:,})")  # .89824297

    r18_spose = corrcoef(squareform(rdm18_spose), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_spose, rdm_2=rdm18_participant, observed_r=r18_spose, n_perms=n_perms)
    print(f"model vs ppts [subset 18]: {r18_spose}; p={p_value} ({n_perms:,})")


    spose_sim48_11 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49_11)
    rdm48_spose_11 = 1 - spose_sim48_11
    spose_sim18_11 = mean_softmax_prob_matrix(all_words=words, select_words=words18_buchanan, full_similarity_matrix=dot_product_49_11)
    rdm18_spose_11 = 1 - spose_sim18_11

    r48_11 = corrcoef(
        squareform(rdm48_spose_11),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose_11, rdm_2=rdm48_participant, observed_r=r48_11, n_perms=n_perms)
    print(f"model[11] vs ppts: {r48_11}; p={p_value} ({n_perms:,})")

    r18_11 = corrcoef(squareform(rdm18_spose_11), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_spose_11, rdm_2=rdm18_participant, observed_r=r18_11, n_perms=n_perms)
    print(f"model[11] vs ppts [subset 18]: {r18_11}; p={p_value} ({n_perms:,})")

    spose_sim48_bottom_11 = mean_softmax_prob_matrix(all_words=words, select_words=words48, full_similarity_matrix=dot_product_49_bottom_11)
    rdm48_spose_bottom_11 = 1 - spose_sim48_bottom_11
    spose_sim18_bottom_11 = mean_softmax_prob_matrix(all_words=words, select_words=words18_buchanan, full_similarity_matrix=dot_product_49_bottom_11)
    rdm18_spose_bottom_11 = 1 - spose_sim18_bottom_11

    r48_bottom_11 = corrcoef(
        squareform(rdm48_spose_bottom_11),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_spose_bottom_11, rdm_2=rdm48_participant, observed_r=r48_bottom_11, n_perms=n_perms)
    print(f"model[bottom_11] vs ppts: {r48_bottom_11}; p={p_value} ({n_perms:,})")

    r18_bottom_11 = corrcoef(squareform(rdm18_spose_bottom_11), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_spose_bottom_11, rdm_2=rdm18_participant, observed_r=r18_bottom_11, n_perms=n_perms)
    print(f"model[bottom_11] vs ppts [subset 18]: {r18_bottom_11}; p={p_value} ({n_perms:,})")

    # endregion

    # region Generate SM RDM for 48 words

    sm = SensorimotorNorms()

    # try and emulate the mean regularised probability from above

    # Get data matrix for all words (that we can)
    sm_data = sm.matrix_for_words(words48)

    sm_rdm_cosine = cosine_distances(sm_data, sm_data)
    sm_sim48_cosine = mean_softmax_prob_matrix(all_words=words48, full_similarity_matrix=1 - sm_rdm_cosine)
    rdm48_sensorimotor_cosine = 1 - sm_sim48_cosine
    sm_sim18_cosine = mean_softmax_prob_matrix(all_words=words48, select_words=words18_buchanan, full_similarity_matrix=1 - sm_rdm_cosine)
    rdm18_sensorimotor_cosine = 1 - sm_sim18_cosine

    sm_r48_cosine = corrcoef(
        squareform(rdm48_sensorimotor_cosine),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_sensorimotor_cosine, rdm_2=rdm48_participant, observed_r=sm_r48_cosine, n_perms=n_perms)
    print(f"sm_cosine vs ppts: {sm_r48_cosine}; p={p_value} ({n_perms:,})")  # .3163023620890325
    print(f"  Without softmax: {corrcoef(squareform(sm_rdm_cosine), squareform(rdm48_participant))[0,1]}")

    sm_r18_cosine = corrcoef(squareform(rdm18_sensorimotor_cosine), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_sensorimotor_cosine, rdm_2=rdm18_participant, observed_r=sm_r18_cosine, n_perms=n_perms)
    print(f"sm_cosine vs ppts [subset 18]: {sm_r18_cosine}; p={p_value} ({n_perms:,})")

    sm_rdm_minkowski = distance_matrix(sm_data, sm_data, p=3)
    sm_sm_minkowski = 1 - (sm_rdm_minkowski / max(sm_rdm_minkowski.flatten()[:])); fill_diagonal(sm_sm_minkowski, 1)
    sm_sim48_minkowski = mean_softmax_prob_matrix(all_words=words48, full_similarity_matrix=sm_sm_minkowski)
    rdm48_sensorimotor_minkowski = 1 - sm_sim48_minkowski
    sm_sim18_minkowski = mean_softmax_prob_matrix(all_words=words48, select_words=words18_buchanan, full_similarity_matrix=sm_sm_minkowski)
    rdm18_sensorimotor_minkowski = 1 - sm_sim18_minkowski

    sm_r48_minkowski = corrcoef(
        squareform(rdm48_sensorimotor_minkowski),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_sensorimotor_minkowski, rdm_2=rdm48_participant, observed_r=sm_r48_minkowski, n_perms=n_perms)
    print(f"sm_minkowski vs ppts: {sm_r48_minkowski}; p={p_value} ({n_perms:,})")  # .2704680432064961
    print(f"  Without softmax: {corrcoef(squareform(sm_rdm_minkowski), squareform(rdm48_participant))[0,1]}")

    sm_r18_minkowski = corrcoef(squareform(rdm18_sensorimotor_minkowski), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_sensorimotor_minkowski, rdm_2=rdm18_participant, observed_r=sm_r18_minkowski, n_perms=n_perms)
    print(f"sm_minkowski vs ppts [subset 18]: {sm_r18_minkowski}; p={p_value} ({n_perms:,})")

    # sm_sm_dotpropduct = dot(sm_data, transpose(sm_data))
    # sm_sim48_dotproduct = mean_softmax_prob_matrix(all_words=words48, full_similarity_matrix=sm_sm_dotpropduct)
    # rdm48_sensorimotor_dotproduct = 1 - sm_sim48_dotproduct
    #
    # sm_r48_dotproduct = corrcoef(
    #     squareform(rdm48_sensorimotor_dotproduct),
    #     squareform(rdm48_participant))[0, 1]
    # p_value = randomisation_p(rdm_1=rdm48_sensorimotor_dotproduct, rdm_2=rdm48_participant, observed_r=sm_r48_dotproduct, n_perms=n_perms)
    # print(f"sm_dotproduct vs ppts: {sm_r48_dotproduct}; p={p_value} ({n_perms:,})")  # .2821415691953627

    wordnet_sm = compute_wordnet_sm(words=words48)
    wordnet_sim48 = mean_softmax_prob_matrix(all_words=words48, full_similarity_matrix=wordnet_sm)
    rdm48_wordnet = 1 - wordnet_sim48
    wordnet_sim18 = mean_softmax_prob_matrix(all_words=words48, select_words=words18_buchanan, full_similarity_matrix=wordnet_sm)
    rdm18_wordnet = 1 - wordnet_sim18

    wordnet_r48 = corrcoef(
        squareform(rdm48_wordnet),
        squareform(rdm48_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm48_wordnet, rdm_2=rdm48_participant, observed_r=wordnet_r48, n_perms=n_perms)
    print(f"wordnet vs ppts: {wordnet_r48}; p={p_value} ({n_perms:,})")
    print(f"  Without softmax: {corrcoef(squareform(1-wordnet_sm), squareform(rdm48_participant))[0,1]}")

    wordnet_r18 = corrcoef(squareform(rdm18_wordnet), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_wordnet, rdm_2=rdm18_participant, observed_r=wordnet_r18, n_perms=n_perms)
    print(f"wordnet vs ppts [subset 18]: {wordnet_r18}; p={p_value} ({n_perms:,})")

    lsa_sm, words46_lsa = compute_lsa_sm(words=words48)
    words46_idxs = find_indices(words48, words46_lsa)
    rdm46_participant = rdm48_participant[ix_(words46_idxs, words46_idxs)]
    lsa_sim46 = mean_softmax_prob_matrix(all_words=words46_lsa, full_similarity_matrix=lsa_sm)
    rdm46_lsa = 1 - lsa_sim46
    lsa_sim18 = mean_softmax_prob_matrix(all_words=words46_lsa, select_words=words18_buchanan, full_similarity_matrix=lsa_sm)
    rdm18_lsa = 1 - lsa_sim18

    lsa_r46 = corrcoef(
        squareform(rdm46_lsa),
        squareform(rdm46_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm46_lsa, rdm_2=rdm46_participant, observed_r=lsa_r46, n_perms=n_perms)
    print(f"lsa vs ppts: {lsa_r46}; p={p_value} ({n_perms:,})")
    print(f"  Without softmax: {corrcoef(squareform(1-lsa_sm), squareform(rdm46_participant))[0,1]}")

    lsa_r18 = corrcoef(squareform(rdm18_lsa), squareform(rdm18_participant))[0, 1]
    p_value = randomisation_p(rdm_1=rdm18_lsa, rdm_2=rdm18_participant, observed_r=lsa_r18, n_perms=n_perms)
    print(f"lsa vs ppts [subset 18]: {lsa_r18}; p={p_value} ({n_perms:,})")

    # endregion

    pass


if __name__ == '__main__':
    main()
