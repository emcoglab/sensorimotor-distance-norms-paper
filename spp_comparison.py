from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Tuple, Optional, Callable

from nltk.corpus import wordnet_ic, wordnet
from numpy import inf
from pandas import DataFrame, read_csv, isna

from linguistic_distributional_models.evaluation.regression import SppData
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType, distance
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

_logger = getLogger(__name__)
_FROM_SCRATCH = True


def add_sensorimotor_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str], distance_type: DistanceType):
    predictor_name = f"Sensorimotor distance ({distance_type.name})"
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
        print_progress(i, n, prefix=f"Sensorimotor {distance_type.name}: ", bar_length=200)
        try:
            v1 = sn.vector_for_word(row[key_col_1])
            v2 = sn.vector_for_word(row[key_col_2])
            return distance(v1, v2, distance_type=distance_type)
        except WordNotInNormsError:
            return None

    # noinspection PyTypeChecker
    dataset[predictor_name] = dataset.apply(calc_sensorimotor_distance, axis=1)


def add_wordnet_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str], pos_columns: Tuple[str, str]):

    # noinspection PyProtectedMember
    from nltk.corpus.reader import WordNetError, NOUN, VERB, ADJ, ADV

    predictor_name = "WordNet distance (JCN)"
    if predictor_name in dataset.columns:
        _logger.info("Predictor already exists, skipping")
        return

    key_col_1, key_col_2 = word_key_cols
    elex_pos_col_1, elex_pos_col_2 = pos_columns

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    elex_to_wordnet_mapping = {
        "nn": NOUN,
        "vb": VERB,
        "jj": ADJ,
        "rb": ADV,
    }

    def elex_to_wordnet(e_pos: str) -> Optional[str]:
        if isna(e_pos):

            return None
        w_pos = [
            elex_to_wordnet_mapping[pos]
            for pos in e_pos.lower().split("|")
            if pos in elex_to_wordnet_mapping
        ]
        try:
            return w_pos[0]
        except IndexError:
            return None

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
        pos_1 = elex_to_wordnet(row[elex_pos_col_1])
        pos_2 = elex_to_wordnet(row[elex_pos_col_2])

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
            pos_columns: Tuple[str, str],
            ):
    _logger.info(out_file_name)

    data_path = Path(out_dir, out_file_name)
    data: DataFrame
    if _FROM_SCRATCH or not data_path.exists():
        _logger.info("Loading from source")
        data = load_from_source()
    else:
        _logger.info("Loading previously saved file")
        with data_path.open(mode="r") as data_file:
            data = read_csv(data_file, header=0, index_col=None)

    _logger.info("Adding WordNet JCN predictor")
    add_wordnet_predictor(data, word_key_cols, pos_columns)
    _logger.info("Saving")
    with data_path.open(mode="w") as out_file:
        data.to_csv(out_file, header=True, index=False)

    _logger.info("Adding sensorimotor predictor")
    add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.Minkowski3)
    add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.cosine)
    # add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.correlation)
    # add_sensorimotor_predictor(data, word_key_cols, distance_type=DistanceType.Euclidean)
    _logger.info("Saving")
    with data_path.open(mode="w") as out_file:
        data.to_csv(out_file, header=True, index=False)

    _logger.info("")


if __name__ == '__main__':

    basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=INFO)

    save_dir = Path("/Users/caiwingfield/Desktop/")

    process(save_dir, "spp.csv",
            lambda: SppData(save_progress=False, force_reload=True).dataframe,
            (SppData.Columns.prime_word, SppData.Columns.target_word),
            pos_columns=(SppData.Columns.prime_pos, SppData.Columns.target_pos))
    _logger.info("Done!")
