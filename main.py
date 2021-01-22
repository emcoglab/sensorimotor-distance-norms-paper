from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Tuple, Optional, Callable

from nltk.corpus import wordnet_ic, wordnet
# noinspection PyProtectedMember
from nltk.corpus.reader import WordNetError, NOUN, VERB, ADJ, ADV
from numpy import inf
from pandas import DataFrame, read_csv

from linguistic_distributional_models.evaluation.association import MenSimilarity, WordsimAll, \
    SimlexSimilarity, WordAssociationTest, RelRelatedness, RubensteinGoodenough, MillerCharlesSimilarity, \
    SmallWorldOfWords
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType, distance
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


_logger = getLogger(__name__)


def load_nelson_data():
    with Path(Path(__file__).parent, "data", "Nelson_AppendixB.csv").open() as nelson_file:
        nelson = read_csv(nelson_file, skip_blank_lines=True, header=0)
    nelson["Targets"] = nelson["Targets"].str.lower()
    nelson["Part of Speech"] = nelson["Part of Speech"].str.lower()
    return nelson


NELSON = load_nelson_data()


def load_jcn_data() -> DataFrame:
    jcn_path = Path(Path(__file__).parent, "data", "Maki-BRMIC-2004", "usfjcnlsa.csv")
    with open(jcn_path) as jcn_file:
        jcn_data: DataFrame = read_csv(jcn_file)
    jcn_data.rename(columns={"#CUE": "CUE"}, inplace=True)
    return jcn_data


def get_nelson_pos(word: str) -> Optional[str]:
    nelson_to_wordnet = {
        "n": NOUN,
        "v": VERB,
        "aj": ADJ,
        "ad": ADV,
    }
    try:
        pos = NELSON[NELSON["Targets"] == word]["Part of Speech"].iloc[0]
    except IndexError:
        # Word not found
        return None
    try:
        return nelson_to_wordnet[pos]
    except KeyError:
        # These are the only POSs we support
        return None


def add_sensorimotor_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str]):
    predictor_name = "Sensorimotor distance"
    if predictor_name in dataset.columns:
        _logger.info("Predictor already exists, skipping")
        return
    key_col_1, key_col_2 = word_key_cols
    sn = SensorimotorNorms()

    i = 0
    n = dataset.shape[0]

    def calc_sensorimotor_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Sensorimotor:          ", bar_length=200)
        try:
            v1 = sn.vector_for_word(row[key_col_1])
            v2 = sn.vector_for_word(row[key_col_2])
            return distance(v1, v2, distance_type=DistanceType.cosine)
        except WordNotInNormsError:
            return None

    # noinspection PyTypeChecker
    dataset[predictor_name] = dataset.apply(calc_sensorimotor_distance, axis=1)


def add_jcn_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str], pos: str):
    predictor_name = "JCN distance"
    if predictor_name in dataset.columns:
        _logger.info("Predictor already exists, skipping")
        return
    key_col_1, key_col_2 = word_key_cols

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    i = 0
    n = dataset.shape[0]

    def calc_jcn_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="WordNet Jiang–Coranth: ", bar_length=200)

        # Get words
        w1 = row[key_col_1]
        w2 = row[key_col_2]

        # Get POS
        if pos.lower() == "nelson":
            pos_1 = get_nelson_pos(w1)
            pos_2 = get_nelson_pos(w2)
            if pos_1 != pos_2:
                # Can only compute distances between word pairs of the same POS
                return None
        else:
            pos_1 = pos_2 = pos

        # Get JCN
        try:
            synsets1 = wordnet.synsets(w1, pos=pos_1)
            synsets2 = wordnet.synsets(w2, pos=pos_2)
        except WordNetError:
            return None
        minimum_jcn_distance = inf
        for synset1 in synsets1:
            for synset2 in synsets2:
                try:
                    jcn = 1 / synset1.jcn_similarity(synset2, brown_ic)  # Match the formula of Maki et al. (2004)
                    minimum_jcn_distance = min(minimum_jcn_distance, jcn)
                except WordNetError:
                    # Skip incomparable pairs
                    continue
                except ZeroDivisionError:
                    # Similarity was zero/distance was infinite
                    continue
        if minimum_jcn_distance >= 1_000:
            return None
        return minimum_jcn_distance

    # noinspection PyTypeChecker
    dataset[predictor_name] = dataset.apply(calc_jcn_distance, axis=1)


def process(out_dir: str,
            out_file_name: str,
            load_from_source: Callable[[], DataFrame],
            word_key_cols: Tuple[str, str],
            pos: Optional[str]):
    _logger.info(out_file_name)

    data_path = Path(out_dir, out_file_name)
    data: DataFrame
    if not data_path.exists():
        _logger.info("Loading from source")
        data = load_from_source()
    else:
        _logger.info("Loading previously saved file")
        with data_path.open(mode="r") as data_file:
            data = read_csv(data_file, header=0, index_col=None)

    _logger.info("Adding sensorimotor predictor")
    add_sensorimotor_predictor(data, word_key_cols)
    _logger.info("Saving")
    with data_path.open(mode="w") as out_file:
        data.to_csv(out_file, header=True, index=False)
    if pos is not None:
        _logger.info("Adding WordNet JCN predictor")
        add_jcn_predictor(data, word_key_cols, pos)
        _logger.info("Saving")
        with data_path.open(mode="w") as out_file:
            data.to_csv(out_file, header=True, index=False)


if __name__ == '__main__':

    basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=INFO)

    save_dir = Path("/Users/caiwingfield/Resilio Sync/Lancaster/CogSci 2021/")

    process(save_dir, "rg.csv", lambda: RubensteinGoodenough().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "n")
    process(save_dir, "miller_charles.csv", lambda: MillerCharlesSimilarity().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "n")
    process(save_dir, "rel.csv", lambda: RelRelatedness().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "n")
    process(save_dir, "wordsim.csv", lambda: WordsimAll().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "nelson")
    process(save_dir, "simlex.csv", lambda: SimlexSimilarity().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "nelson")
    process(save_dir, "men.csv", lambda: MenSimilarity().associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "nelson")
    process(save_dir, "jcn.csv", lambda: load_jcn_data(), ("CUE", "TARGET"), "nelson")
    process(save_dir, "swow_r1.csv", lambda: SmallWorldOfWords(responses_type=SmallWorldOfWords.ResponsesType.R1).associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "n")
    process(save_dir, "swow_r123.csv", lambda: SmallWorldOfWords(responses_type=SmallWorldOfWords.ResponsesType.R123).associations_to_dataframe(), (WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), "n")

    _logger.info("Done!")
