from logging import getLogger
from typing import Dict

# noinspection PyProtectedMember
from nltk.corpus.reader import NOUN, VERB, ADJ, ADV

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Convert from Elexicon POS tags to Wordnet POS tags
elex_to_wordnet: Dict[str, str] = {
    "nn": NOUN,
    "vb": VERB,
    "jj": ADJ,
    "rb": ADV,
}
