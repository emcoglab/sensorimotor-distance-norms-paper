from logging import basicConfig, INFO
from pathlib import Path
from random import seed
from typing import Tuple, Iterable

from pandas import DataFrame, concat, read_csv
from scipy.io import loadmat
from scipy.spatial.distance import cdist

from linguistic_distributional_models.evaluation.association import WordsimAll, WordAssociationTest, SimlexSimilarity, \
    MenSimilarity
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType
from predictors.aux import logger, logger_format, logger_dateformat, pos_dir, lsa_dir, hebart_dir
from predictors.spose import SPOSE
from predictors.predictors import add_wordnet_predictor, add_lsa_predictor, add_norms_overlap_predictor, \
    add_sensorimotor_predictor
from predictors.rsa import compute_buchanan_sm, RDM, SimilarityMatrix, compute_lsa_sm, compute_wordnet_sm, \
    compute_sensorimotor_rdm, subset_flag
from predictors.wordnet import WordnetAssociation
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from visualisation.distributions import graph_distance_distribution


def common_similarity_modelling(df: DataFrame,
                                word_key_cols: Tuple[str, str],
                                pos_path: Path,
                                lsa_path: Path,
                                save_path: Path):
    """
    Use all predictors to model a given similarity judgement dataset.

    :param df:
    :param word_key_cols:
    :param pos_path:
    :param lsa_path:
    :param save_path:
    :return:
    """

    logger.info("Adding predictors")
    df = add_wordnet_predictor(
        df,
        word_key_cols=word_key_cols,
        pos_path=pos_path,
        association_type=WordnetAssociation.JiangConrath)
    df = add_lsa_predictor(
        df,
        word_key_cols=word_key_cols,
        lsa_path=lsa_path)
    df = add_norms_overlap_predictor(
        df,
        word_key_cols=word_key_cols)
    df = add_sensorimotor_predictor(
        df,
        word_key_cols=word_key_cols,
        distance_type=DistanceType.cosine)

    logger.info(f"Saving results to {save_path}")
    with save_path.open("w") as save_file:
        df.to_csv(save_file, header=True, index=False)

    return df


def model_wordsim(location: Path, overwrite: bool) -> DataFrame:
    """
    Model the WordSim dataset using all predictors.

    :param location:
    :param overwrite:
    :return:
    """

    save_path = Path(location, "wordsim.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        with save_path.open("r") as f:
            return read_csv(f)

    wordsim_data = common_similarity_modelling(
        # Load data from source:
        df=WordsimAll().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(lsa_dir, "wordsim-lsa.csv"),
        pos_path=Path(pos_dir, "wordsim-pos.tab"),
        save_path=save_path)

    return wordsim_data


def model_simlex(location: Path, overwrite: bool) -> DataFrame:
    """
    Model the Simlex dataset using all predictors.

    :param location:
    :param overwrite:
    :return:
    """

    save_path = Path(location, "simlex.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        with save_path.open("r") as f:
            return read_csv(f)

    simlex_data = common_similarity_modelling(
        # Load data from source:
        df=SimlexSimilarity().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(lsa_dir, "simlex-lsa.csv"),
        pos_path=Path(pos_dir, "simlex-pos.tab"),
        save_path=save_path)

    return simlex_data


def model_men(location: Path, overwrite: bool) -> DataFrame:
    """
    Model the MEN dataset using all predictors.

    :param location:
    :param overwrite:
    :return:
    """
    save_path = Path(location, "men.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        with save_path.open("r") as f:
            return read_csv(f)

    men_data = common_similarity_modelling(
        # Load data from source:
        df=MenSimilarity().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(lsa_dir, "men-lsa.csv"),
        pos_path=Path(pos_dir, "men-pos.tab"),
        save_path=save_path)

    return men_data


def save_raw_values(reference_rdm: RDM, comparison_rdm: RDM, save_path: Path, overwrite: bool):
    """
    Saves the raw LTV values from two RDMs into a 3-column CSV file so they can be analysed with other software
    (e.g. JASP). The columns are: values from reference RDM; values from comparison RDM; binary flag which is True just
    when both (original matrix's) row and columns were in the common set of 18 words.

    :param reference_rdm:
    :param comparison_rdm:
    :param save_path:
    :param overwrite:
    :return:
    """
    assert reference_rdm.labels == comparison_rdm.labels

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        return

    save_path.parent.mkdir(exist_ok=True)
    with save_path.open("w") as save_file:
        DataFrame.from_dict({
            "Reference RDM values": reference_rdm.triangular_values,
            "Comparison RDM values": comparison_rdm.triangular_values,
            "In common 18": subset_flag(reference_rdm, subset_labels=SPOSE.words_common_18).triangular_values,
        }).to_csv(save_file, header=True, index=False)


def model_hebart(location: Path, overwrite: bool, n_perms: int) -> None:
    """
    Model the Hebart et al. participant RDMs using a suite of predictors. In each case, use their method of modelling
    the choice-probability similarity matrix using a mean softmax transform of the predictor's similarity matrix.

    :param location: Where to save the results. Raw LTV RDM values will be saved so that JASP (etc.) can be used to
        perform more rigorous statistical testing.
    :param overwrite: Whether to overwrite existing results.
    :param n_perms: How many permutations to use for nonparametric NHST.
    """

    results_path = Path(location, "hebart_results.csv")

    # TODO: (minor) this doesn't really respect the "overwrite" flag properly as it will prevent raw values being saved
    #  below.
    if results_path.exists() and not overwrite:
        logger.warning(f"{results_path} exists, skipping")
        return

    logger.info(f"Hebart modelling ({n_perms:,} permutations)")

    results = []

    # Reference RDM
    rdm_participants = RDM(matrix=loadmat(Path(hebart_dir, "RDM48_triplet.mat"))["RDM48_triplet"],
                           labels=SPOSE.words_select_48)

    # Hebart's SPOSE embedding
    rdm_spose = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(
            SimilarityMatrix.by_dotproduct(data_matrix=SPOSE.matrix_49d, labels=SPOSE.words_all),
            subset_labels=SPOSE.words_select_48))
    rdm_spose_top11 = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(
            SimilarityMatrix.by_dotproduct(data_matrix=SPOSE.matrix_49d[:, :11], labels=SPOSE.words_all),
            subset_labels=SPOSE.words_select_48))
    rdm_spose_bottom11 = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(
            SimilarityMatrix.by_dotproduct(data_matrix=SPOSE.matrix_49d[:, -11:], labels=SPOSE.words_all),
            subset_labels=SPOSE.words_select_48))
    results.extend([
        ("Participants", "SPOSE",
         48, *rdm_participants
         .correlate_with_nhst(rdm_spose, n_perms=n_perms)),
        ("Participants", "SPOSE",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_spose.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
        ("Participants", "SPOSE top-11",
         48, *rdm_participants
         .correlate_with_nhst(rdm_spose_top11, n_perms=n_perms)),
        ("Participants", "SPOSE top-11",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_spose_top11.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
        ("Participants", "SPOSE bottom-11",
         48, *rdm_participants
         .correlate_with_nhst(rdm_spose_bottom11, n_perms=n_perms)),
        ("Participants", "SPOSE bottom-11",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_spose_bottom11.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
    ])
    save_raw_values(rdm_participants, rdm_spose, Path(location, "spose.csv"), overwrite)
    save_raw_values(rdm_participants, rdm_spose_top11, Path(location, "spose_top11.csv"), overwrite)
    save_raw_values(rdm_participants, rdm_spose_bottom11, Path(location, "spose_bottom11.csv"), overwrite)

    # LSA
    rdm_lsa = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(compute_lsa_sm(), SPOSE.words_lsa_46))
    results.extend([
        ("Participants", "LSA softmax",
         46, *rdm_participants.for_subset(SPOSE.words_lsa_46)
         .correlate_with_nhst(rdm_lsa, n_perms=n_perms)),
        ("Participants", "LSA softmax",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_lsa.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
    ])
    save_raw_values(rdm_participants.for_subset(SPOSE.words_lsa_46), rdm_lsa, Path(location, "lsa.csv"), overwrite)

    # Wordnet
    rdm_wordnet_jcn = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(compute_wordnet_sm(association_type=WordnetAssociation.JiangConrath)))
    rdm_wordnet_res = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(compute_wordnet_sm(association_type=WordnetAssociation.Resnik)))
    results.extend([
        ("Participants", f"Wordnet {WordnetAssociation.JiangConrath.name}",
         48, *rdm_participants
         .correlate_with_nhst(rdm_wordnet_jcn, n_perms=n_perms)),
        ("Participants", f"Wordnet {WordnetAssociation.JiangConrath.name}",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_wordnet_jcn.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
    ])
    results.extend([
        ("Participants", f"Wordnet {WordnetAssociation.Resnik.name}",
         48, *rdm_participants
         .correlate_with_nhst(rdm_wordnet_res, n_perms=n_perms)),
        ("Participants", f"Wordnet {WordnetAssociation.Resnik.name}",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_wordnet_res.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
    ])
    save_raw_values(rdm_participants, rdm_wordnet_jcn, Path(location, f"wordnet {WordnetAssociation.JiangConrath.name}.csv"), overwrite)
    save_raw_values(rdm_participants, rdm_wordnet_res, Path(location, f"wordnet {WordnetAssociation.Resnik.name}.csv"), overwrite)

    # Buchanan
    rdm_buchanan = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(compute_buchanan_sm(), SPOSE.words_common_18))
    results.extend([
        ("Participants", "Buchanan feature overlap",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_buchanan, n_perms=n_perms))
    ])
    save_raw_values(rdm_participants.for_subset(SPOSE.words_common_18), rdm_buchanan, Path(location, "buchanan.csv"), overwrite)

    # Sensorimotor
    sensorimotor_distance = DistanceType.cosine
    rdm_sensorimotor = RDM.from_similarity_matrix(
        SimilarityMatrix.mean_softmax_probability_matrix(
            SimilarityMatrix.from_rdm(compute_sensorimotor_rdm(distance_type=sensorimotor_distance))))
    results.extend([
        ("Participants", f"Sensorimotor {sensorimotor_distance.name}",
         48, *rdm_participants
         .correlate_with_nhst(rdm_sensorimotor, n_perms=n_perms)),
        ("Participants", f"Sensorimotor {sensorimotor_distance.name}",
         18, *rdm_participants.for_subset(SPOSE.words_common_18)
         .correlate_with_nhst(rdm_sensorimotor.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
    ])
    save_raw_values(rdm_participants, rdm_sensorimotor, Path(location, "sensorimotor.csv"), overwrite)

    # Sensorimotor 10-dim subspaces
    for exclude_dimension in SensorimotorNorms().VectorColNames:
        rdm_sensorimotor = RDM.from_similarity_matrix(
            SimilarityMatrix.mean_softmax_probability_matrix(
                SimilarityMatrix.from_rdm(compute_sensorimotor_rdm(distance_type=sensorimotor_distance,
                                                                   exclude_dimension=exclude_dimension))))
        results.extend([
            ("Participants", f"Sensorimotor {sensorimotor_distance.name} excluding {exclude_dimension}",
             48, *rdm_participants
             .correlate_with_nhst(rdm_sensorimotor, n_perms=n_perms)),
            ("Participants", f"Sensorimotor {sensorimotor_distance.name} excluding {exclude_dimension}",
             18, *rdm_participants.for_subset(SPOSE.words_common_18)
             .correlate_with_nhst(rdm_sensorimotor.for_subset(SPOSE.words_common_18), n_perms=n_perms)),
        ])
        save_raw_values(rdm_participants, rdm_sensorimotor,
                        Path(location, f"sensorimotor_excluding_{exclude_dimension}.csv"), overwrite)

    with results_path.open("w") as save_file:
        DataFrame.from_records(
            results,
            columns=[
                "Comparison LH",
                "Comparison RH",
                "N conditions",
                "R-value",
                f"P-value ({n_perms:_} permutations)",
            ]
        ).to_csv(save_file, header=True, index=False)


def save_combined_pairs(dfs: Iterable[DataFrame], location: Path) -> None:
    combined_data: DataFrame = concat(dfs)
    combined_data.drop_duplicates(subset=[WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2],
                                  inplace=True)
    # This isn't comparable between datasets, and isn't used for the gramd correlation
    del combined_data[WordAssociationTest.TestColumn.association_strength]

    with Path(location, "combined.csv").open("w") as f:
        combined_data.to_csv(f, header=True, index=False)


def save_full_pairwise_distances(location: Path, overwrite: bool):
    WORD_1 = "Word 1"
    WORD_2 = "Word 2"
    DISTANCE = "Distance"

    sensorimotor_norms = SensorimotorNorms()

    logger.info("Computing all distance pairs")

    final_filename = "all_distances.csv"
    if Path(location, final_filename).exists() and not overwrite:
        logger.info(f"File {final_filename} exists, skipping.")
        return

    temporary_csv_path = Path(location, f"{final_filename}.incomplete")

    all_words = sorted(sensorimotor_norms.iter_words())

    with temporary_csv_path.open("w") as temp_file:
        # Write the header
        temp_file.write(f"{WORD_1},{WORD_2},{DISTANCE}")
        for word_i, word in enumerate(all_words):

            # Only the LTV, don't double-count (diagonal is fine)
            other_words = all_words[word_i:]

            # Pairwise distances for this word and all other (not yet seen, i.e. ltv-only) words
            distances = cdist(
                sensorimotor_norms.vector_for_word(word).reshape(1, -1),
                sensorimotor_norms.matrix_for_words(other_words),
                'cosine').reshape(-1)

            these_distances = DataFrame()
            these_distances[WORD_2] = other_words
            these_distances[DISTANCE] = distances
            these_distances[WORD_1] = word

            # Append this block of distances
            these_distances[[WORD_1, WORD_2, DISTANCE]].to_csv(temp_file, header=False, index=False)

            print_progress(word_i, len(all_words))
    print_progress(len(all_words), len(all_words))

    # Move into place
    temporary_csv_path.rename(Path(location, final_filename))


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)

    # For reproducibility
    seed(1)

    # TODO: make these CLI args
    save_dir = Path("/Users/caiwingfield/Box Sync/LANGBOOT Project/Manuscripts/Draft - Sensorimotor distance norms/Output/")
    overwrite = True

    # Graph distributions for each measure
    figures_location = Path(save_dir, "Figures")
    n_bins = 20
    ylim = None
    graph_distance_distribution(
        distance_type=DistanceType.cosine, n_bins=n_bins, location=figures_location, overwrite=overwrite, ylim=ylim)
    graph_distance_distribution(
        distance_type=DistanceType.correlation, n_bins=n_bins, location=figures_location, overwrite=overwrite, ylim=ylim)
    graph_distance_distribution(
        distance_type=DistanceType.Minkowski3, n_bins=n_bins, location=figures_location, overwrite=overwrite, ylim=ylim)
    graph_distance_distribution(
        distance_type=DistanceType.Euclidean, n_bins=n_bins, location=figures_location, overwrite=overwrite, ylim=ylim)

    # Run each of the analyses in turn
    wordsim_data = model_wordsim(location=save_dir, overwrite=overwrite)
    simlex_data  = model_simlex(location=save_dir, overwrite=overwrite)
    men_data     = model_men(location=save_dir, overwrite=overwrite)

    save_combined_pairs((wordsim_data, simlex_data, men_data), location=save_dir)

    # n_perms = 100_000
    # model_hebart(location=Path(save_dir, "Hebart"), overwrite=overwrite, n_perms=n_perms)

    save_full_pairwise_distances(location=Path("/Users/caiwingfield/Desktop"), overwrite=overwrite)

    logger.info("Done!")
