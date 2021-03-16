from enum import Enum, auto
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from nltk.corpus import wordnet_ic, wordnet
# noinspection PyProtectedMember
from nltk.corpus.reader import WordNetError
from numpy import inf
from pandas import DataFrame, read_csv, merge

from aux import logger, elex_to_wordnet
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import distance, DistanceType
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

_brown_ic = wordnet_ic.ic('ic-brown.dat')


class WordnetAssociation(Enum):
    """Representative of a type of Wordnet distance."""

    JCN = auto()  # JCN distance a la Maki et al. (2004)
    Resnik = auto()  # Resnik similarity

    @property
    def name(self):
        if self == self.JCN:
            return "Jiang-Coranth"
        if self == self.Resnik:
            return "Resnik"
        raise NotImplementedError()

    def association_between(self, word_1, word_2, word_1_pos, word_2_pos) -> Optional[float]:
        """
        :param word_1, word_2: The words
        :param word_1_pos, word_2_pos: The words' respective parts of speech tags
        :return: The association value, or None if at least one of the words wasn't available.
        """
        try:
            synsets_1 = wordnet.synsets(word_1, pos=word_1_pos)
            synsets_2 = wordnet.synsets(word_2, pos=word_2_pos)

            if self == self.Resnik:
                max_similarity = 0
                for s1 in synsets_1:
                    for s2 in synsets_2:
                        max_similarity = max(max_similarity, s1.res_similarity(s2, _brown_ic))
                return max_similarity

            if self == self.JCN:
                minimum_jcn_distance = inf
                for s1 in synsets_1:
                    for s2 in synsets_2:
                        try:
                            minimum_jcn_distance = min(
                                minimum_jcn_distance,
                                # Match the formula of Maki et al. (2004)
                                1 / s1.jcn_similarity(s2, _brown_ic))
                        except WordNetError:
                            continue  # Skip incomparable pairs
                        except ZeroDivisionError:
                            continue  # Similarity was zero
                # Catch cases where we're still at inf
                if minimum_jcn_distance >= 100_000:
                    return None
                return minimum_jcn_distance

            raise NotImplementedError()

        except WordNetError:
            return None


class BuchananFeatureNorms:

    class SingleWordsCols:
        Cue = "CUE"

    class DoubleWordsCols:
        Cue = "CUE"
        Target = "TARGET"
        CosineRoot = "root"

    _data_dir = Path(Path(__file__).parent, "data", "buchanan")
    _single_words_path = Path(_data_dir, "single_word.csv")
    _double_words_path = Path(_data_dir, "double_words.csv")

    def __init__(self):
        with self._double_words_path.open("r") as double_words_file:
            self.double_words_data: DataFrame = read_csv(double_words_file)
        with self._single_words_path.open("r") as single_words_file:
            self.single_words_data: DataFrame = read_csv(single_words_file)

        self.available_words: List[str] = sorted(set(self.single_words_data[BuchananFeatureNorms.SingleWordsCols.Cue]))

    def distance_between(self, word_1, word_2):
        word_1, word_2 = tuple(sorted((word_1, word_2)))
        if word_1 not in self.available_words:
            raise KeyError(word_1)
        if word_2 not in self.available_words:
            raise KeyError(word_2)
        try:
            return self.double_words_data.loc[
                (self.double_words_data[BuchananFeatureNorms.DoubleWordsCols.Cue] == word_1)
                & (self.double_words_data[BuchananFeatureNorms.DoubleWordsCols.Target] == word_2)
                ][BuchananFeatureNorms.DoubleWordsCols.CosineRoot].values[0]
        except IndexError:
            # We have verified already that the word queries are good, so if the pair is missing, it has a similarity of
            # zero
            return 0


sensorimotor_norms = SensorimotorNorms()
buchanan_feature_norms = BuchananFeatureNorms()


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
        If a column named "LSA" already exists in `df`, nothing will be added.
    """
    _predictor_name = "LSA"

    if _predictor_name in df.columns:
        logger.warning("Predictor already exists, skipping")
        return df

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
        If the column already existed in `df`, nothing new will be added.
    """
    _predictor_name = f"WordNet association ({association_type.name})"

    if _predictor_name in df.columns:
        logger.info("Predictor already exists, skipping")
        return

    assert len(word_key_cols) == 2
    key_col_1, key_col_2 = word_key_cols

    brown_ic = wordnet_ic.ic('ic-brown.dat')

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

        return WordnetAssociation.JCN.association_between(
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
        If a column of the same name already existed, nothing will be added.
    """
    _predictor_name = "Buchanan root cosine overlap"

    if _predictor_name in df.columns:
        logger.info("Predictor already exists, skipping")
        return

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
            return buchanan_feature_norms.distance_between(row[key_col_1], row[key_col_2])
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
            v1 = sensorimotor_norms.vector_for_word(row[key_col_1])
            v2 = sensorimotor_norms.vector_for_word(row[key_col_2])
            return distance(v1, v2, distance_type=distance_type)
        except WordNotInNormsError:
            return None

    # noinspection PyTypeChecker
    df[_predictor_name] = df.apply(calc_sensorimotor_distance, axis=1)

    return df
