from pathlib import Path
from typing import Tuple, Dict, Optional

# noinspection PyProtectedMember
from pandas import DataFrame, read_csv, merge, concat

from linguistic_distributional_models.utils.logging import print_progress
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from .aux import logger
from .distance import Distance, Cosine
from .mandera import MANDERA_CBOW
from .wordnet import elex_to_wordnet
from .buchanan import BUCHANAN_FEATURE_NORMS
from .wordnet import WordnetAssociation

_sensorimotor_norms = SensorimotorNorms()


class PredictorName:

    @staticmethod
    def lsa():
        return "LSA"

    @staticmethod
    def mandera_cbow():
        return "Mandera CBOW (cosine)"

    @staticmethod
    def wordnet(for_association_type: WordnetAssociation):
        return f"WordNet association ({for_association_type.name})"

    @staticmethod
    def feature_overlap():
        return "Buchanan root cosine overlap"

    @staticmethod
    def sensorimotor_distance(for_distance: Distance):
        return f"Sensorimotor distance ({for_distance.name})"

    @staticmethod
    def sensory_distance(for_distance: Distance):
        return f"Perceptual distance ({for_distance.name})"

    @staticmethod
    def motor_distance(for_distance: Distance):
        return f"Action distance ({for_distance.name})"


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

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    with lsa_path.open("r") as lsa_file:
        lsa_deets: DataFrame = read_csv(lsa_file, header=None)
    lsa_deets.columns = [key_col_1, key_col_2, PredictorName.lsa()]
    # Don't know which way the pair goes, so add the flipped versions
    lsa_deets_swapped = lsa_deets.copy()
    lsa_deets_swapped.columns = [key_col_2, key_col_1, PredictorName.lsa()]
    lsa_deets = concat([lsa_deets, lsa_deets_swapped], ignore_index=True, join="inner")

    # Duplicated rows causes the left merge to behave badly, so ensure we don't have any
    lsa_deets.drop_duplicates(subset=[key_col_1, key_col_2], inplace=True)

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

    def calc_wordnet_distance(row):
        if association_type != WordnetAssociation.JiangConrath:
            raise NotImplementedError()

        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"{PredictorName.wordnet(association_type)}: ")

        # Get words
        w1 = row[key_col_1]
        w2 = row[key_col_2]

        return WordnetAssociation.JiangConrath.distance_between(
            word_1=w1, word_1_pos=get_pos(w1),
            word_2=w2, word_2_pos=get_pos(w2),
        )

    # noinspection PyTypeChecker
    df[PredictorName.wordnet(association_type)] = df.apply(calc_wordnet_distance, axis=1)

    return df


def add_mandera_predictor(df: DataFrame, word_key_cols: Tuple[str, str]):
    """
    Adds a column of Mandera's recommended CBOW cosine distances to the supplied dataframe

    :param df:
        The reference DataFrame
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which feature overlap will be
        calculated.
    :return:
        `df` plus an appropriately named feature overlap column.
    """

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    # For capture by next closure
    i = 0
    n = df.shape[0]

    def calc_mandera_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"Mandera CBOW cosine")

        # Get words
        w1 = row[key_col_1]
        w2 = row[key_col_2]

        return MANDERA_CBOW.distance_between(w1, w2, distance=Cosine())

    df[PredictorName.mandera_cbow()] = df.apply(calc_mandera_distance, axis=1)

    return df


def add_feature_overlap_predictor(df: DataFrame, word_key_cols: Tuple[str, str]):
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

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    # To be captured by following closure
    i = 0
    n = df.shape[0]

    def calc_feature_overlap(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix=f"Buchanan overlap: ")
        try:
            return BUCHANAN_FEATURE_NORMS.overlap_between(row[key_col_1], row[key_col_2])
        except KeyError:
            return None

    # noinspection PyTypeChecker
    df[PredictorName.feature_overlap()] = df.apply(calc_feature_overlap, axis=1)

    return df


def add_sensorimotor_predictor(df: DataFrame, word_key_cols: Tuple[str, str], distance: Distance, only: Optional[str] = None):
    """
    Adds a column of sensorimotor disatnces to the specified dataframe.

    :param df:
        The reference DataFrame.
    :param word_key_cols:
        2-tuple of strings: the column names for the first and second words in a pair for which sensorimotor distances
        will be calculated.
    :param distance:
        The `Distance` to use when computing sensorimotor distances.
    :return:
        `df` plus an appropriately named sensorimotor distance column added.
        If a column of the same name already existed, nothing will be added.
    """

    if only is None:
        predictor_name = PredictorName.sensorimotor_distance(distance)
    elif only == "sensory":
        predictor_name = PredictorName.sensory_distance(distance)
    elif only == "motor":
        predictor_name = PredictorName.motor_distance(distance)
    else:
        raise ValueError()

    if predictor_name in df.columns:
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
        if only is None:
            prfx = "Sensorimotor"
            sensory_only = motor_only = None
        elif only == "sensory":
            prfx = "Sensory"
            sensory_only = True
            motor_only = False
        elif only == "motor":
            prfx = "Motor"
            sensory_only = False
            motor_only = True
        else:
            raise ValueError()
        print_progress(i, n, prefix=f"{prfx} {distance.name}: ")
        try:
            if sensory_only:
                v1 = _sensorimotor_norms.sensory_vector_for_word(row[key_col_1].lower())
                v2 = _sensorimotor_norms.sensory_vector_for_word(row[key_col_2].lower())
            elif motor_only:
                v1 = _sensorimotor_norms.motor_vector_for_word(row[key_col_1].lower())
                v2 = _sensorimotor_norms.motor_vector_for_word(row[key_col_2].lower())
            else:
                v1 = _sensorimotor_norms.sensorimotor_vector_for_word(row[key_col_1].lower())
                v2 = _sensorimotor_norms.sensorimotor_vector_for_word(row[key_col_2].lower())
        except WordNotInNormsError:
            return None
        return distance.distance(v1, v2)

    # noinspection PyTypeChecker
    df[predictor_name] = df.apply(calc_sensorimotor_distance, axis=1)

    return df
