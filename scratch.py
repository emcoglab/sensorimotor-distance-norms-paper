from pathlib import Path

from pandas import read_csv

from linguistic_distributional_models.evaluation.association import SimlexSimilarity, WordAssociationTest


def scratch():
    add_concreteness_ratings(SimlexSimilarity().associations_to_dataframe()).to_csv("/Users/caiwingfield/Desktop/simlex_conc.csv", index=False)


def add_concreteness_ratings(association_df):
    concreteness_df = read_csv(
        Path(Path(__file__).parent, "data", "concreteness", "13428_2013_403_MOESM1_ESM.csv").as_posix())
    association_df = association_df.merge(concreteness_df.rename(
        columns={"Word": WordAssociationTest.TestColumn.word_1, "Conc.M": "Word 1 concreteness"})[
                                    [WordAssociationTest.TestColumn.word_1, "Word 1 concreteness"]], how="left",
                                          on=WordAssociationTest.TestColumn.word_1)
    association_df = association_df.merge(concreteness_df.rename(
        columns={"Word": WordAssociationTest.TestColumn.word_2, "Conc.M": "Word 2 concreteness"})[
                                    [WordAssociationTest.TestColumn.word_2, "Word 2 concreteness"]], how="left",
                                          on=WordAssociationTest.TestColumn.word_2)
    return association_df


if __name__ == '__main__':
    scratch()
