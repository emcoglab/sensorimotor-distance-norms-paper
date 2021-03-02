from __future__ import annotations
from pathlib import Path
from typing import List

from pandas import DataFrame, read_csv


class SingleWordsCols:
    Cue = "CUE"


class DoubleWordsCols:
    Cue = "CUE"
    Target = "TARGET"
    CosineRoot = "root"


class BuchananFeatureNorms:
    _data_dir = Path(Path(__file__).parent, "data", "buchanan")
    _single_words_path = Path(_data_dir, "single_word.csv")
    _double_words_path = Path(_data_dir, "double_words.csv")

    def __init__(self):
        with self._double_words_path.open("r") as double_words_file:
            self.double_words_data: DataFrame = read_csv(double_words_file)
        with self._single_words_path.open("r") as single_words_file:
            self.single_words_data: DataFrame = read_csv(single_words_file)

        self.available_words: List[str] = sorted(set(self.single_words_data[SingleWordsCols.Cue]))

    def distance_between(self, word_1, word_2):
        word_1, word_2 = tuple(sorted((word_1, word_2)))
        if word_1 not in self.available_words:
            raise KeyError(word_1)
        if word_2 not in self.available_words:
            raise KeyError(word_2)
        try:
            return self.double_words_data.loc[
                (self.double_words_data[DoubleWordsCols.Cue] == word_1)
                & (self.double_words_data[DoubleWordsCols.Target] == word_2)
                ][DoubleWordsCols.CosineRoot].values[0]
        except IndexError:
            # We have verified already that the word queries are good, so if the pair is missing, it has a similarity of
            # zero
            return 0


if __name__ == '__main__':
    print(BuchananFeatureNorms().distance_between("abandon", "boop"))
