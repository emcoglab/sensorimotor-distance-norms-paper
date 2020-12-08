from pathlib import Path

from numpy import log
from pandas import DataFrame, Series

from linguistic_distributional_models.corpus.indexing import FreqDist
from linguistic_distributional_models.preferences.preferences import Preferences as LDMPreferences
from linguistic_distributional_models.evaluation.association import WordAssociationTest, MenSimilarity
from linguistic_distributional_models.model.predict import CbowModel
from linguistic_distributional_models.utils.exceptions import WordNotFoundError
from linguistic_distributional_models.utils.logging import print_progress
from linguistic_distributional_models.utils.maths import DistanceType, distance
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


corpus = LDMPreferences.source_corpus_metas.ukwac


def load_dataset() -> DataFrame:
    wordsim_associations: DataFrame = MenSimilarity().associations_to_dataframe()
    return wordsim_associations


def add_linguistic_predictor(dataset: DataFrame):
    distance_measure = DistanceType.cosine
    model = CbowModel(corpus_meta=corpus,
                      window_radius=10,
                      embedding_size=200)
    model.train(memory_map=True)
    i = 0
    n = dataset.shape[0]

    def calc_linguistic_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Adding linguistic predictor: ")
        try:
            return model.distance_between(
                row[WordAssociationTest.TestColumn.word_1],
                row[WordAssociationTest.TestColumn.word_2],
                distance_type=distance_measure)
        except WordNotFoundError:
            return None

    dataset["Linguistic distance"] = dataset.apply(calc_linguistic_distance, axis=1)


def add_sensorimotor_predictor(dataset: DataFrame):
    sn = SensorimotorNorms()
    i = 0
    n = dataset.shape[0]

    def calc_sensorimotor_distance(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Adding sensorimotor predictor: ")
        try:
            v1 = sn.vector_for_word(row[WordAssociationTest.TestColumn.word_1])
            v2 = sn.vector_for_word(row[WordAssociationTest.TestColumn.word_2])
            return distance(v1, v2, distance_type=DistanceType.cosine)
        except WordNotInNormsError:
            return None

    dataset["Sensorimotor distance"] = dataset.apply(calc_sensorimotor_distance, axis=1)


def add_ancillary_predictors(dataset: DataFrame):
    fd = FreqDist.load(corpus.freq_dist_path)

    i = 0
    n = dataset.shape[0]

    def calc_ancillary_predictors(row):
        nonlocal i
        i += 1
        print_progress(i, n, prefix="Adding ancillary predictors: ")

        word_1 = row[WordAssociationTest.TestColumn.word_1]
        word_2 = row[WordAssociationTest.TestColumn.word_2]

        return Series({
            "Word 1 length": len(word_1),
            "Word 2 length": len(word_2),
            "Word 1 frequency": fd[word_1],
            "Word 2 frequency": fd[word_2],
            "Log Word 1 frequency": log(fd[word_1] + 1),
            "Log Word 2 frequency": log(fd[word_2] + 1),
        })

    dataset[[
        "Word 1 length",
        "Word 2 length",
        "Word 1 frequency",
        "Word 2 frequency",
        "Log Word 1 frequency",
        "Log Word 2 frequency",
    ]] = dataset.apply(calc_ancillary_predictors, axis=1)



def main():
    out_path = Path("/Users/caiwingfield/Desktop/test.csv")

    dataset: DataFrame = load_dataset()
    add_ancillary_predictors(dataset)
    add_linguistic_predictor(dataset)
    add_sensorimotor_predictor(dataset)
    with open(out_path, mode="w", encoding="utf-8") as out_file:
        dataset.to_csv(out_file, header=True, index=False)


if __name__ == '__main__':
    main()
