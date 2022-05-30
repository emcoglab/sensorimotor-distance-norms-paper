from pathlib import Path

from pandas import DataFrame, read_csv

from predictors.aux import mandera_dir
from predictors.distance import Distance


class ManderaCBOW:
    _data_path_cbow = Path(mandera_dir, "english-all.words-cbow-window.6-dimensions.300-ukwac_subtitle_en.w2v")

    def __init__(self):
        with self._data_path_cbow.open("r") as data_file:
            self.data: DataFrame = read_csv(data_file, header=3, index_col=0, delim_whitespace=True)

    def distance_between(self, w1: str, w2: str, distance: Distance):
        if w1 not in self.data.index:
            return None
        if w2 not in self.data.index:
            return None
        return distance.distance(self.data.loc[w1], self.data.loc[w2])


# shared copy
MANDERA_CBOW: ManderaCBOW = ManderaCBOW()
