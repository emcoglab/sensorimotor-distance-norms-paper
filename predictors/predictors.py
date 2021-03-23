from pathlib import Path
from typing import Tuple, Dict, Optional

# noinspection PyProtectedMember
from pandas import DataFrame, read_csv, merge

from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import distance, DistanceType
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from .aux import logger
from .wordnet import elex_to_wordnet
from .buchanan import BUCHANAN_FEATURE_NORMS
from .wordnet import WordnetAssociation

_sensorimotor_norms = SensorimotorNorms()


def add_lsa_predictor(df: DataFrame, word_key_cols: Tuple[str, str], lsa_path: Path) -> DataFrame:
    """
    Adds a column of LSA distances to the specified dataframe.

    :param df:
        The reference dataframe
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which LSA values will be
        looked up.
    :param lsa_path:
        `Path` to the location of the file containing the LSA distances.
        This must be a 3-column csv
    :return:
        `df`, plus the "LSA" column.
    """
    _predictor_name = "LSA"

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    with lsa_path.open("r") as lsa_file:
        lsa_deets: DataFrame = read_csv(lsa_file, header=None)
    lsa_deets.columns = [key_col_1, key_col_2, _predictor_name]

    df = merge(df, lsa_deets, on=[key_col_1, key_col_2], how="left")
    return df


def add_wordnet_predictor(df: DataFrame,
                          word_key_cols: Tuple[str, str],
                          pos_path: Optional[Path],
                          association_type: WordnetAssociation):
    """
    Adds a column of Wordnet distance predictor to the supplied dataframe.

    :param df:
        The reference dataframe
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which wordnet distance values
        will be computed.
    :param pos_path:
        Path to a two-column tab-separated file containing a row for each word and elexicon-coded POS tags.
    :param association_type:
        Which type of wordnet association to use
    :return:
        `df` with the appropriately named wordneet distance column added.
    """
    _predictor_name = f"WordNet association ({association_type.name})"

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    elex_pos: Dict[str, str]
    if pos_path is not None:
        with pos_path.open("r") as pos_file:
            elex_df = read_csv(pos_file, header=0, index_col=None, delimiter="\t")
            elex_df.set_index("Word", inplace=True)
            elex_dict: dict = elex_df.to_dict('index')
            elex_pos = {
                word: [
                    elex_to_wordnet[pos.lower()]
                    for pos in data["POS"].split("|")
                    if pos in elex_to_wordnet
                ]
                for word, data in elex_dict.items()
            }
    else:
        elex_pos = dict()

    def get_pos(word) -> Optional[str]:
        if elex_pos is None:
            return None
        try:
            # Assume Elexicon lists POS in precedence order
            return elex_pos[word][0]
        except KeyError:
            return None
        except IndexError:
            return None

    # For capture by next closure
    i = 0
    n = df.shape[0]

    def calc_jcn_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"WordNet {association_type.name}: ", bar_length=200)

        # Get words
        w1 = row[key_col_1]
        w2 = row[key_col_2]

        return WordnetAssociation.JiangConrath.distance_between(
            word_1=w1, word_1_pos=get_pos(w1),
            word_2=w2, word_2_pos=get_pos(w2),
        )

    # noinspection PyTypeChecker
    df[_predictor_name] = df.apply(calc_jcn_distance, axis=1)

    return df


def add_norms_overlap_predictor(df: DataFrame, word_key_cols: Tuple[str, str]):
    """
    Adds a column of feature-overlap measures from the Buchanan feature norms to the supplied dataframe.

    :param df:
        The reference DataFrame
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which feature overlap will be
        calculated.
    :return:
        `df` plus an appropriately named feature overlap column.
    """
    _predictor_name = "Buchanan root cosine overlap"

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    # To be captured by following closure
    i = 0
    n = df.shape[0]

    def calc_norms_overlap(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"Buchanan overlap: ", bar_length=200)
        try:
            return BUCHANAN_FEATURE_NORMS.overlap_between(row[key_col_1], row[key_col_2])
        except KeyError:
            return None

    # noinspection PyTypeChecker
    df[_predictor_name] = df.apply(calc_norms_overlap, axis=1)

    return df


def add_sensorimotor_predictor(df: DataFrame, word_key_cols: Tuple[str, str], distance_type: DistanceType):
    """
    Adds a column of sensorimotor disatnces to the specified dataframe.

    :param df:
        The reference DataFrame.
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which sensorimotor distances
        will be calculated.
    :param distance_type:
        The `DistanceType` to use when computing sensorimotor distances.
    :return:
        `df` plus an appropriately named sensorimotor distance column added.
        If a column of the same name already existed, nothing will be added.
    """
    _predictor_name = f"Sensorimotor distance ({distance_type.name})"

    if _predictor_name in df.columns:
        logger.info("Predictor already exists, skipping")
        return

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    # To be captured by following closure
    i = 0
    n = df.shape[0]

    def calc_sensorimotor_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"Sensorimotor {distance_type.name}: ", bar_length=200)
        try:
            v1 = _sensorimotor_norms.vector_for_word(row[key_col_1])
            v2 = _sensorimotor_norms.vector_for_word(row[key_col_2])
            return distance(v1, v2, distance_type=distance_type)
        except WordNotInNormsError:
            return None

    # noinspection PyTypeChecker
    df[_predictor_name] = df.apply(calc_sensorimotor_distance, axis=1)

    return df
