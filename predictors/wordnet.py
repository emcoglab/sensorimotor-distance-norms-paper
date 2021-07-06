from enum import Enum, auto
from typing import Optional, Dict

from nltk.corpus import wordnet_ic, wordnet
from nltk.corpus.reader import WordNetError, NOUN, VERB, ADJ, ADV

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

    JiangConrath = auto()  # JCN distance à la Jiang & Conrath (1997) and Maki et al. (2004)
    Resnik = auto()  # Resnik similarity

    @property
    def name(self):
        if self == self.JiangConrath:
            return "Jiang–Conrath"
        if self == self.Resnik:
            return "Resnik"
        raise NotImplementedError()

    def distance_between(self, word_1, word_2, word_1_pos, word_2_pos) -> Optional[float]:
        """
        Compute the specified similarity between pos-tagged words.

        :param word_1, word_2: The words
        :param word_1_pos, word_2_pos: The words' respective parts of speech tags
        :return: The association value, or None if at least one of the words wasn't available.
        """
        if self == self.JiangConrath:
            similarity = self.similarity_between(word_1, word_2, word_1_pos, word_2_pos)
            if similarity is None:
                return None
            # Avoid divide-by-zero errors
            if similarity == 0:
                return None
            # Avoid nearly divide-by-zero errors
            if similarity < 0.000001:
                return None
            # Match the formula of Maki et al. (2004)
            return 1 / similarity
        else:
            raise NotImplementedError()

    def similarity_between(self, word_1, word_2, word_1_pos, word_2_pos) -> Optional[float]:
        """
        Compute the specified similarity between pos-tagged words.

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

            if self == self.JiangConrath:
                maximum_jcn_similarity = 0
                for s1 in synsets_1:
                    for s2 in synsets_2:
                        try:
                            maximum_jcn_similarity = max(
                                maximum_jcn_similarity,
                                s1.jcn_similarity(s2, _brown_ic))
                        # Skip incomparable pairs
                        except WordNetError:
                            continue
                return maximum_jcn_similarity

            raise NotImplementedError()

        except WordNetError:
            return None
