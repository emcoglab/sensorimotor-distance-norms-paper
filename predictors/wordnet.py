from enum import Enum, auto
from typing import Optional, Dict

from nltk.corpus import wordnet_ic, wordnet
from nltk.corpus.reader import WordNetError, NOUN, VERB, ADJ, ADV
from numpy import inf

_brown_ic = wordnet_ic.ic('ic-brown.dat')

# Convert from Elexicon POS tags to Wordnet POS tags
elex_to_wordnet: Dict[str, str] = {
    "nn": NOUN,
    "vb": VERB,
    "jj": ADJ,
    "rb": ADV,
}


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
        Compute the specified distance between pos-tagged words.

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
