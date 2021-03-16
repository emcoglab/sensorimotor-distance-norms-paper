from logging import basicConfig, INFO
from pathlib import Path
from typing import Tuple

from pandas import DataFrame

from aux import logger, logger_format, logger_dateformat
from linguistic_distributional_models.utils.maths import DistanceType
from predictors import add_wordnet_predictor, WordnetDistance, add_lsa_predictor, add_norms_overlap_predictor, \
    add_sensorimotor_predictor
from linguistic_distributional_models.evaluation.association import WordsimAll, WordAssociationTest, SimlexSimilarity, \
    MenSimilarity

_pos_dir = Path(Path(__file__).parent, "data", "elexicon")
_lsa_dir = Path(Path(__file__).parent, "data", "LSA")


def common_similarity_modelling(df: DataFrame,
                                word_key_cols: Tuple[str, str],
                                pos_path: Path,
                                lsa_path: Path,
                                save_path: Path):
    logger.info("Adding predictors")
    df = add_wordnet_predictor(
        df,
        word_key_cols=word_key_cols,
        pos_path=pos_path,
        distance_type=WordnetDistance.JCN)
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


def model_wordsim(location: Path, overwrite: bool) -> None:
    save_path = Path(location, "wordsim.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        return

    common_similarity_modelling(
        # Load data from source:
        df=WordsimAll().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(_lsa_dir, "wordsim-lsa.csv"),
        pos_path=Path(_pos_dir, "wordsim-pos.tab"),
        save_path=save_path)


def model_simlex(location: Path, overwrite: bool) -> None:
    save_path = Path(location, "simlex.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        return

    common_similarity_modelling(
        # Load data from source:
        df=SimlexSimilarity().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(_lsa_dir, "simlex-lsa.csv"),
        pos_path=Path(_pos_dir, "simlex-pos.tab"),
        save_path=save_path)


def model_men(location: Path, overwrite: bool) -> None:
    save_path = Path(location, "men.csv")

    if save_path.exists() and not overwrite:
        logger.warning(f"{save_path} exists, skipping")
        return

    common_similarity_modelling(
        # Load data from source:
        df=MenSimilarity().associations_to_dataframe(),
        word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
        lsa_path=Path(_lsa_dir, "men-lsa.csv"),
        pos_path=Path(_pos_dir, "men-pos.tab"),
        save_path=save_path)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)

    save_dir = Path("/Users/caiwingfield/Resilio Sync/Lancaster/Sensorimotor distance paper/Output/")
    overwrite = True

    model_wordsim(location=save_dir, overwrite=overwrite)
    model_simlex(location=save_dir, overwrite=overwrite)
    model_men(location=save_dir, overwrite=overwrite)

    logger.info("Done!")
