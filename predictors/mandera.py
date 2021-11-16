from pathlib import Path

from pandas import DataFrame, read_csv

from linguistic_distributional_models.utils.maths import DistanceType, distance
from predictors.aux import mandera_dir


class ManderaCBOW:
    _data_path_cbow = Path(mandera_dir, "english-all.words-cbow-window.6-dimensions.300-ukwac_subtitle_en.w2v")

    def __init__(self):
        with self._data_path_cbow.open("r") as data_file:
            self.data: DataFrame = read_csv(data_file, header=3, index_col=0, delim_whitespace=True)

    def distance_between(self, w1: str, w2: str, distance_type: DistanceType):
        if w1 not in self.data.index:
            return None
        if w2 not in self.data.index:
            return None
        return distance(self.data.loc[w1], self.data.loc[w2], distance_type=distance_type)


# shared copy
MANDERA_CBOW: ManderaCBOW = ManderaCBOW()
