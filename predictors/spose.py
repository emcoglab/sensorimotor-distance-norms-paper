from pathlib import Path

from numpy import array
from pandas import read_csv
from scipy.io import loadmat

from .aux import hebart_dir


class Spose:
    def __init__(self):
        # Data
        with Path(hebart_dir, "spose_embedding_49d_sorted.txt").open() as spose_49_sorted_file:
            self.matrix_49d: array = read_csv(
                spose_49_sorted_file,
                skip_blank_lines=True, header=None, delimiter=" "
            ).to_numpy()[:, :49]

        # Words
        self.words_all = [w[0][0] for w in loadmat(Path(hebart_dir, "words.mat"))["words"]]
        self.words_select_48  = [w[0][0] for w in loadmat(Path(hebart_dir, "words48.mat"))["words48"]]
        self.words_lsa_46 = Path(hebart_dir, "words46lsa.txt").read_text().split("\n")
        self.words_common_18 = Path(hebart_dir, "words18common.txt").read_text().split("\n")


SPOSE = Spose()
