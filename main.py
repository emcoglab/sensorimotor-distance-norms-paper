from pathlib import Path
from typing import Tuple, Optional

from nltk.corpus import wordnet_ic, wordnet
from nltk.corpus.reader import WordNetError
from pandas import DataFrame, read_csv

from linguistic_distributional_models.evaluation.association import MenSimilarity, WordsimAll, \
    SimlexSimilarity, WordAssociationTest, RelRelatedness, RubensteinGoodenough, MillerCharlesSimilarity
from linguistic_distributional_models.evaluation.regression import SppData
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType, distance
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


def load_men_data() -> DataFrame:
    men_associations: DataFrame = MenSimilarity().associations_to_dataframe()
    return men_associations


def load_wordsim_data() -> DataFrame:
    wordsim_associations: DataFrame = WordsimAll().associations_to_dataframe()
    return wordsim_associations


def load_simlex_data() -> DataFrame:
    wordsim_associations: DataFrame = SimlexSimilarity().associations_to_dataframe()
    return wordsim_associations


def load_priming_data() -> DataFrame:
    priming_data: DataFrame = SppData().dataframe
    return priming_data


def load_rel_data() -> DataFrame:
    wordsim_associations: DataFrame = RelRelatedness().associations_to_dataframe()
    return wordsim_associations


def load_rg65_data() -> DataFrame:
    rg_associations: DataFrame = RubensteinGoodenough().associations_to_dataframe()
    return rg_associations


def load_miller_charles_data() -> DataFrame:
    mc_similarity: DataFrame = MillerCharlesSimilarity().associations_to_dataframe()
    return mc_similarity


def load_jcn_data() -> DataFrame:
    jcn_path = Path(Path(__file__).parent, "data", "Maki-BRMIC-2004", "usfjcnlsa.csv")
    with open(jcn_path) as jcn_file:
        jcn_data: DataFrame = read_csv(jcn_file)
    jcn_data.rename(columns={"#CUE": "CUE"}, inplace=True)
    return jcn_data


def add_extra_predictors(dataset: DataFrame, word_key_cols: Tuple[str, str], pos: Optional[str] = None):
    add_sensorimotor_predictor(dataset, word_key_cols)
    if pos is not None:
        add_jcn_predictor(dataset, word_key_cols, pos)


def add_sensorimotor_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str]):
    key_col_1, key_col_2 = word_key_cols
    sn = SensorimotorNorms()

    i = 0
    n = dataset.shape[0]

    def calc_sensorimotor_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Adding sensorimotor predictor: ")
        try:
            v1 = sn.vector_for_word(row[key_col_1])
            v2 = sn.vector_for_word(row[key_col_2])
            return distance(v1, v2, distance_type=DistanceType.cosine)
        except WordNotInNormsError:
            return None

    dataset["Sensorimotor distance"] = dataset.apply(calc_sensorimotor_distance, axis=1)


def load_nelson_data():
    with Path(Path(__file__).parent, "data", "Nelson_AppendixB.csv").open() as nelson_file:
        nelson = read_csv(nelson_file, skip_blank_lines=True, header=0)
    nelson["Targets"] = nelson["Targets"].str.lower()
    nelson["Part of Speech"] = nelson["Part of Speech"].str.lower()
    return nelson


nelson = load_nelson_data()


def add_jcn_predictor(dataset: DataFrame, word_key_cols: Tuple[str, str], pos: str):
    key_col_1, key_col_2 = word_key_cols

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    i = 0
    n = dataset.shape[0]

    def calc_jcn_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Adding Jiang–Coranth predictor: ")

        # Get words
        w1 = row[key_col_1]
        w2 = row[key_col_2]

        # Get POS
        if pos != "nelson":
            pos_1 = pos_2 = pos
        else:
            try:
                pos_1 = nelson[nelson["Targets"] == w1]["Part of Speech"].iloc[0]
                pos_2 = nelson[nelson["Targets"] == w2]["Part of Speech"].iloc[0]
            except IndexError:
                return None

        # Get JCN
        try:
            synset1 = wordnet.synset(f"{w1}.{pos_1}.01")
            synset2 = wordnet.synset(f"{w2}.{pos_2}.01")
            jcn = 1 / synset1.jcn_similarity(synset2, brown_ic)  # To match the fomula used by Maki et al. (2004)
            if jcn >= 1_000:
                return None
            return jcn
        except WordNetError:
            return None

    dataset["JCN distance"] = dataset.apply(calc_jcn_distance, axis=1)


def main():
    out_dir = Path("/Users/caiwingfield/Desktop/")

    # Load data
    data_wordsim: DataFrame = load_wordsim_data()
    data_simlex: DataFrame = load_simlex_data()
    data_men: DataFrame = load_men_data()
    data_priming: DataFrame = load_priming_data()
    data_rel: DataFrame = load_rel_data()
    data_rg: DataFrame = load_rg65_data()
    data_mc: DataFrame = load_miller_charles_data()
    data_jcn: DataFrame = load_jcn_data()

    # Add sensorimotor predictors
    add_extra_predictors(data_wordsim, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="nelson")
    add_extra_predictors(data_simlex, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="nelson")
    add_extra_predictors(data_men, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="nelson")
    add_extra_predictors(data_priming, word_key_cols=(SppData.Columns.prime_word, SppData.Columns.target_word), pos="nelson")
    add_extra_predictors(data_rel, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="n")
    add_extra_predictors(data_rg, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="n")
    add_extra_predictors(data_mc, word_key_cols=(WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2), pos="n")
    add_extra_predictors(data_jcn, ("CUE", "TARGET"), pos="nelson")

    # Save
    with open(Path(out_dir, "wordsim.csv"), mode="w", encoding="utf-8") as out_file:
        data_wordsim.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "simlex.csv"), mode="w", encoding="utf-8") as out_file:
        data_simlex.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "men.csv"), mode="w", encoding="utf-8") as out_file:
        data_men.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "priming.csv"), mode="w", encoding="utf-8") as out_file:
        data_priming.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "rel.csv"), mode="w", encoding="utf-8") as out_file:
        data_rel.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "rg.csv"), mode="w", encoding="utf-8") as out_file:
        data_rg.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "miller_charles.csv"), mode="w", encoding="utf-8") as out_file:
        data_mc.to_csv(out_file, header=True, index=False)
    with open(Path(out_dir, "jcn.csv"), mode="w", encoding="utf-8") as out_file:
        data_jcn.to_csv(out_file, header=True, index=False)


if __name__ == '__main__':
    main()
