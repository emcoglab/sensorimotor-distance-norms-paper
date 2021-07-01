from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

# noinspection PyProtectedMember
from nltk.corpus.reader import NOUN
from numpy import zeros, corrcoef, ix_, exp, fill_diagonal, transpose, array, dot
from numpy.ma import array as masked_array
from numpy.random import permutation
from pandas import read_csv
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_distances

from linguistic_distributional_models.utils.maths import DistanceType
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from .aux import find_indices, lsa_dir
from .buchanan import BUCHANAN_FEATURE_NORMS
from .spose import SPOSE
from .wordnet import WordnetAssociation


class LabelledSymmetricMatrix:
    """
    A symmetric matrix with a set of labels referring to rows and columns.
    """
    def __init__(self, matrix: array, labels: List[str]):
        self.matrix: array = matrix
        self.labels: List[str] = labels

        assert len(labels) == matrix.shape[0] == matrix.shape[1]
        assert (matrix == transpose(matrix)).all()

    @property
    def triangular_values(self) -> array:
        """Just the values in the lower (or upper) triangular vector."""
        return squareform(self.matrix, checks=False)

    def for_subset(self, subset_words: List[str]) -> LabelledSymmetricMatrix:
        """Matrix for a subset of the rows and columns."""
        idxs = find_indices(self.labels, subset_words)
        assert len(idxs) == len(subset_words)
        return LabelledSymmetricMatrix(
            matrix=self.matrix[ix_(idxs, idxs)],
            labels=subset_words)

    def correlate_with(self, other: LabelledSymmetricMatrix) -> float:
        """Pearson's correlation of the triangular values with another matrix."""
        assert len(self.labels) == len(other.labels)
        return corrcoef(self.triangular_values, other.triangular_values)[0, 1]


class SimilarityMatrix(LabelledSymmetricMatrix):
    """
    A labelled similarity matrix.
    """

    # Make sure return type is correct
    def for_subset(self, subset_words: List[str]) -> SimilarityMatrix:
        s = super().for_subset(subset_words)
        return SimilarityMatrix(
            matrix=s.matrix,
            labels=s.labels
        )

    @classmethod
    def by_dotproduct(cls, data_matrix: array, labels: List[str]) -> SimilarityMatrix:
        """
        Generate a similarity matrix using dot product on rows of a data matrix.

        :param data_matrix:
        :param labels:
        :return:
        """
        return cls(
            matrix=dot(data_matrix, transpose(data_matrix)),
            labels=labels)

    @classmethod
    def mean_softmax_probability_matrix(cls,
                                        from_similarity_matrix: SimilarityMatrix,
                                        subset_labels: Optional[List[str]] = None
                                        ) -> SimilarityMatrix:
        """
        Convert a similarity matrix to a mean-softmax probability matrix using the method of Hebart et al.

        :param from_similarity_matrix:
        :param subset_labels:
        :return:
        """

        if subset_labels is None:
            subset_labels = from_similarity_matrix.labels
        idxs = find_indices(from_similarity_matrix.labels, subset_labels)
        assert len(idxs) == len(subset_labels)  # make sure we're not missing anything

        exp_similarity_matrix = exp(from_similarity_matrix.matrix)
        cp = zeros((len(from_similarity_matrix.labels), len(from_similarity_matrix.labels)))
        # Hebart et al.'s original code builds the entire matrix for all conditions, then selects out the relevant
        # entries. We can hugely speed up this process by only computing the entries we'll eventually select out.
        for i in idxs:
            # print_progress(i, n_all_conditions, prefix=prefix)
            for j in idxs:
                if i == j: continue
                ctmp = zeros((1, len(from_similarity_matrix.labels)))
                for k in idxs:
                    # Only interested in distinct triplets
                    if (k == i) or (k == j):
                        continue
                    ctmp[0, k] = (
                            exp_similarity_matrix[i, j]
                            / (
                                    exp_similarity_matrix[i, j]
                                    + exp_similarity_matrix[i, k]
                                    + exp_similarity_matrix[j, k]
                            ))
                cp[i, j] = ctmp.sum()
        # print_progress(n_all_conditions, n_all_conditions, prefix=prefix)
        # Complete average
        cp /= len(subset_labels)
        # Fill in the rest of the symmetric similarity matrix
        # cp += transpose(cp)  # No longer need to do this now we're filling in both sides of the matrix in the above loop
        # Instead we fix rounding errors by forcing symmetry
        cp += transpose(cp); cp /= 2
        fill_diagonal(cp, 1)
        # Select out words of interest
        selected_similarities = cp[ix_(idxs, idxs)]
        return SimilarityMatrix(matrix=selected_similarities, labels=subset_labels)

    @staticmethod
    def from_rdm(rdm: RDM) -> SimilarityMatrix:
        """
        Convert an RDM to a similarity matrix.
        :param rdm:
        :return:
        """
        return SimilarityMatrix(matrix=1-rdm.matrix, labels=rdm.labels)


class RDM(LabelledSymmetricMatrix):
    """
    A labelled representational dissimilarity matrix (RDM).
    """

    # Make sure return type is correct
    def for_subset(self, subset_words: List[str]) -> RDM:
        s = super().for_subset(subset_words)
        return RDM(
            matrix=s.matrix,
            labels=s.labels
        )

    @staticmethod
    def from_similarity_matrix(similarity_matrix: SimilarityMatrix) -> RDM:
        """
        Get an RDM from a similarity matrix.
        :param similarity_matrix:
        :return:
        """
        return RDM(matrix=1 - similarity_matrix.matrix, labels=similarity_matrix.labels)

    def correlate_with_nhst(self, other: LabelledSymmetricMatrix, n_perms: int) -> Tuple[float, float]:
        """
        Correlation, using conditional-label randomisation test to generate a p-value.

        :param other:
        :param n_perms:
        :return: r-value, p-value
        """
        r_value = self.correlate_with(other)
        p_value = randomisation_p(rdm_1=self.matrix, rdm_2=other.matrix, observed_r=r_value, n_perms=n_perms)
        return r_value, p_value


def randomisation_p(rdm_1, rdm_2, observed_r, n_perms):
    """
    Compute a p-value by randomisation test.

    Under H0, condition labels are exchangeable. Simulate null distribution of r-values by permuting labels of one RDM
    and recomputing r. Then the p-value is the fraction of the distribution above the observed r.

    :param rdm_1, rdm_2: The two RDMs to be correlated
    :param observed_r: The correlation already observed
    :param n_perms: The number of prmutations to perform
    :return:
    """
    r_perms = zeros(n_perms)
    c1 = squareform(rdm_1)
    for perm_i in range(n_perms):
        # if perm_i % 1000 == 0: print(perm_i)
        perm = permutation(rdm_1.shape[0])
        r_perms[perm_i] = corrcoef(
            c1,
            squareform(rdm_2[ix_(perm, perm)])
        )[0, 1]
    p_value = 1 - (percentileofscore(r_perms, observed_r) / 100)
    return p_value


def compute_wordnet_sm(association_type: WordnetAssociation):
    """
    Compute a similarity matrix on Hebart et al's 48 select words using wordnet association.

    :param association_type:
    :return:
    """
    n_words = len(SPOSE.words_select_48)
    similarity_matrix = zeros((n_words, n_words))
    for i in range(n_words):
        for j in range(n_words):
            similarity_matrix[i, j] = association_type.similarity_between(
                word_1=SPOSE.words_select_48[i], word_1_pos=NOUN,
                word_2=SPOSE.words_select_48[j], word_2_pos=NOUN)
    fill_diagonal(similarity_matrix, 1)
    return SimilarityMatrix(matrix=similarity_matrix, labels=SPOSE.words_select_48)


def compute_lsa_sm():
    """
    Compute a similarity matrix on 46 of Hebart et al's 48 select words using LSA.

    :return:
    """
    similarity_matrix_df = read_csv(Path(lsa_dir, "hebart48-lsa.csv"), header=0, index_col=0)
    similarity_matrix = similarity_matrix_df[SPOSE.words_lsa_46].loc[SPOSE.words_lsa_46].to_numpy(dtype=float)
    return SimilarityMatrix(matrix=similarity_matrix, labels=SPOSE.words_lsa_46)


def compute_buchanan_sm():
    """
    Compute a similarity matrix on 18 of Hebart et al's 48 select words using Buchanan feature overlap.

    :return:
    """
    n_words = len(SPOSE.words_common_18)
    similarity_matrix = zeros((n_words, n_words))
    for i in range(n_words):
        for j in range(n_words):
            similarity_matrix[i, j] = BUCHANAN_FEATURE_NORMS.overlap_between(SPOSE.words_common_18[i], SPOSE.words_common_18[j])
    fill_diagonal(similarity_matrix, 1)
    return SimilarityMatrix(matrix=similarity_matrix, labels=SPOSE.words_common_18)


def compute_sensorimotor_rdm(distance_type, exclude_dimension: Optional[str] = None) -> RDM:
    """
    Compute a RDM matrix using sensorimotor distance on Hebart et al.'s 48 select words.

    :param distance_type:
    :param exclude_dimension:
        To assess the relative contributions of each of the sensorimotor dimensions to this, optionally exclude one of
        them by name. Pass None (the default_ to not exclude a dimension.
    :return:
    """

    sm_data = SensorimotorNorms().matrix_for_words(SPOSE.words_select_48)
    if exclude_dimension is not None:
        # Mask and then exclude the entries from the excluded dimension
        excluded_dimension_idx = SensorimotorNorms().VectorColNames.index(exclude_dimension)
        if excluded_dimension_idx == -1:
            raise ValueError(exclude_dimension)
        masked_sm_data = masked_array(sm_data, mask=False)
        masked_sm_data.mask[:, excluded_dimension_idx] = True
        sm_data = masked_sm_data.compressed().reshape((48, 10))
    if distance_type == DistanceType.cosine:
        rdm = cosine_distances(sm_data)
    elif distance_type == DistanceType.Minkowski3:
        rdm = distance_matrix(sm_data, sm_data, p=3)
    else:
        raise NotImplementedError()

    return RDM(matrix=rdm, labels=SPOSE.words_select_48)


def subset_flag(reference_rdm, subset_labels) -> SimilarityMatrix:
    """
    Given a subset of labels, returns a similarity matrix whose entries are True where both row and column labels are
    within the subset and otherwise False.

    :param reference_rdm:
    :param subset_labels:
    :return:
    """
    flag = zeros((len(reference_rdm.labels), len(reference_rdm.labels)), dtype=bool)
    idxs = find_indices(reference_rdm.labels, subset_labels)
    flag[ix_(idxs, idxs)] = True
    return SimilarityMatrix(matrix=flag, labels=reference_rdm.labels)
