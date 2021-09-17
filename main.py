from logging import basicConfig, INFO
from pathlib import Path
from random import seed
from typing import Tuple, Iterable

from pandas import DataFrame, concat, read_csv, isna
from scipy.spatial.distance import cdist

from linguistic_distributional_models.evaluation.association import WordsimAll, WordAssociationTest, SimlexSimilarity, \
    MenSimilarity
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType
from predictors.aux import logger, logger_format, logger_dateformat, pos_dir, lsa_dir
from predictors.predictors import add_wordnet_predictor, add_lsa_predictor, add_feature_overlap_predictor, \
    add_sensorimotor_predictor, PredictorName
from predictors.wordnet import WordnetAssociation
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from visualisation.distributions import graph_distance_distribution


def common_similarity_modelling(df: DataFrame,
                                word_key_cols: Tuple[str, str],
                                dv_col: str,
                                pos_path: Path,
                                lsa_path: Path,
                                save_path: Path):
    """
    Use all predictors to model a given similarity judgement dataset.

    :param df:
    :param word_key_cols:
    :param dv_col:
    :param pos_path:
    :param lsa_path:
    :param save_path:
    :return:
    """

    df.rename(columns={WordAssociationTest.TestColumn.association_strength: dv_col}, inplace=True)
    logger.info("Adding concreteness labels")
    df = label_with_concreteness(df)
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
    df = add_feature_overlap_predictor(
        df,
        word_key_cols=word_key_cols)
    df = add_sensorimotor_predictor(
        df,
        word_key_cols=word_key_cols,
        distance_type=DistanceType.cosine)
    df = add_sensorimotor_predictor(
        df,
        word_key_cols=word_key_cols,
        distance_type=DistanceType.correlation)
    df = add_sensorimotor_predictor(
        df,
        word_key_cols=word_key_cols,
        distance_type=DistanceType.Euclidean)
    df = add_sensorimotor_predictor(
        df,
        word_key_cols=word_key_cols,
        distance_type=DistanceType.Minkowski3)
    df["Common to all predictors"] = (
        ~isna(df[PredictorName.wordnet(WordnetAssociation.JiangConrath)])
        & ~isna(df[PredictorName.lsa()])
        & ~isna(df[PredictorName.feature_overlap()])
        & ~isna(df[PredictorName.sensorimotor_distance(DistanceType.cosine)])  # Identical for all DistanceTypes
    )

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
        dv_col = "WordSim similarity judgement",
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
        dv_col="Simlex similarity judgement",
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
        dv_col="MEN similarity judgement",
        lsa_path=Path(lsa_dir, "men-lsa.csv"),
        pos_path=Path(pos_dir, "men-pos.tab"),
        save_path=save_path)

    return men_data


def save_combined_pairs(dfs: Iterable[DataFrame], location: Path) -> None:
    combined_data: DataFrame = concat(dfs,
                                      # The DV columns aren't comparable between datasets, and can't be used for the
                                      # grand correlation. However they all have different names so we can just exclude
                                      # them automatically.
                                      join='inner')
    combined_data.drop_duplicates(subset=[WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2],
                                  inplace=True)

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


def label_with_concreteness(df: DataFrame) -> DataFrame:

    concreteness_df = read_csv(
        Path(Path(__file__).parent, "data", "concreteness", "13428_2013_403_MOESM1_ESM.csv").as_posix())

    # Get first-word-in-pair concreteness rating
    df = df.merge(
        concreteness_df
            .rename(columns={"Word": WordAssociationTest.TestColumn.word_1, "Conc.M": "Word 1 concreteness"})
            [[WordAssociationTest.TestColumn.word_1, "Word 1 concreteness"]],
        how="left", on=WordAssociationTest.TestColumn.word_1)
    # Get second-word-in-pair concreteness rating
    df = df.merge(
        concreteness_df
            .rename(columns={"Word": WordAssociationTest.TestColumn.word_2, "Conc.M": "Word 2 concreteness"})
        [[WordAssociationTest.TestColumn.word_2, "Word 2 concreteness"]],
        how="left", on=WordAssociationTest.TestColumn.word_2)

    def pair_type(row) -> str:
        # Concrete defined to be >=3 on the 1–5 Brysbaert scale. <3 is therefore abstract
        word_1_is_concrete: bool = row["Word 1 concreteness"] >= 3
        word_2_is_concrete: bool = row["Word 2 concreteness"] >= 3
        if word_1_is_concrete and word_2_is_concrete:
            return "concrete_concrete"
        if (not word_1_is_concrete) and (not word_2_is_concrete):
            return "abstract_abstract"
        return"mixed"

    df["Pair type"] = df.apply(pair_type, axis=1)

    return df


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

    # This makes a 30GB file so don't run it unless you really want to
    # save_full_pairwise_distances(location=save_dir, overwrite=overwrite)

    logger.info("Done!")
