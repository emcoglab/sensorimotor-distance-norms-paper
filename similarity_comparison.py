from logging import basicConfig, INFO
from pathlib import Path
from typing import Tuple, Optional, Callable

from pandas import DataFrame, read_csv

from aux import logger
from linguistic_distributional_models.evaluation.association import MenSimilarity, WordsimAll, \
    SimlexSimilarity, WordAssociationTest
from linguistic_distributional_models.utils.maths import DistanceType
from predictors import add_lsa_predictor, add_wordnet_predictor, WordnetDistance, add_sensorimotor_predictor, \
    add_norms_overlap_predictor

_FROM_SCRATCH = True


def process(out_dir: str,
            out_file_name: str,
            load_from_source: Callable[[], DataFrame],
            word_key_cols: Tuple[str, str],
            pos_filename: Optional[str],
            lsa_filename: Optional[str],
            ):
    logger.info(out_file_name)

    data_path = Path(out_dir, out_file_name)
    data: DataFrame
    if _FROM_SCRATCH or not data_path.exists():
        logger.info("Loading from source")
        data = load_from_source()
    else:
        logger.info("Loading previously saved file")
        with data_path.open(mode="r") as data_file:
            data = read_csv(data_file, header=0, index_col=None)

    if pos_filename is not None:
        logger.info("Adding WordNet JCN predictor")
        add_wordnet_predictor(data, word_key_cols, Path(Path(__file__).parent, "data", "elexicon", pos_filename),
                              distance_type=WordnetDistance.JCN)
        logger.info("Saving")
        with data_path.open(mode="w") as out_file:
            data.to_csv(out_file, header=True, index=False)

    if lsa_filename is not None:
        logger.info("Adding LSA predictor")
        data = add_lsa_predictor(data, word_key_cols, Path(Path(__file__).parent, "data", "LSA", lsa_filename))
        logger.info("Saving")
        with data_path.open(mode="w") as out_file:
            data.to_csv(out_file, header=True, index=False)

    logger.info("Adding Buchanan feature norms overlap predictor")
    add_norms_overlap_predictor(data, word_key_cols)

    logger.info("Adding sensorimotor predictor")
    add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.Minkowski3)
    add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.cosine)

    logger.info("Saving")
    with data_path.open(mode="w") as out_file:
        data.to_csv(out_file, header=True, index=False)


if __name__ == '__main__':

    basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=INFO)

    save_dir = Path("/Users/caiwingfield/Resilio Sync/Lancaster/CogSci 2021/")
    process(save_dir, "wordsim.csv",
            lambda: WordsimAll().associations_to_dataframe(),
            (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
            pos_filename="wordsim-pos.tab",
            lsa_filename="wordsim-lsa.csv")
    process(save_dir, "simlex.csv",
            lambda: SimlexSimilarity().associations_to_dataframe(),
            (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
            pos_filename="simlex-pos.tab",
            lsa_filename="simlex-lsa.csv")
    process(save_dir, "men.csv",
            lambda: MenSimilarity().associations_to_dataframe(),
            (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2),
            pos_filename="men-pos.tab",
            lsa_filename="men-lsa.csv")

    logger.info("Done!")
