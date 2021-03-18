from logging import getLogger

# noinspection PyProtectedMember
from pathlib import Path

from numpy import array, searchsorted


# Logging
logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

# Relative paths
_data_dir = Path(Path(__file__).parent.parent, "data")
pos_dir = Path(_data_dir, "elexicon")
lsa_dir = Path(_data_dir, "LSA")
hebart_dir = Path(_data_dir, "hebart")
buchanan_dir = Path(_data_dir, "buchanan")


def find_indices(super_list, sub_list):
    """
    Finds indices of all elements of sub_list in super_list, in order.

    :param super_list:
    :param sub_list:
    :return:
    """
    for sub in sub_list:
        assert sub in super_list
    sort_idx = array(super_list).argsort()
    positions_selected = sort_idx[searchsorted(super_list, sub_list, sorter=sort_idx)]
    return positions_selected
