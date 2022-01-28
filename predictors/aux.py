from logging import getLogger

# noinspection PyProtectedMember
from pathlib import Path


# Logging
logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Relative paths
_data_dir = Path(Path(__file__).parent.parent, "data")
pos_dir = Path(_data_dir, "elexicon")
lsa_dir = Path(_data_dir, "LSA")
buchanan_dir = Path(_data_dir, "buchanan")
mandera_dir = Path(_data_dir, "Mandera")
