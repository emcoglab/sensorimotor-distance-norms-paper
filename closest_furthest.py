"""
===========================
Find closest and furthest pairs in the whole set of norms.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2022
---------------------------
"""


from pathlib import Path

from numpy import inf

all_ds = Path("/Users/caiwingfield/Box Sync/LANGBOOT Project/Manuscripts/Draft - Sensorimotor distance norms/Output/")

min_dist = inf
min_w1, min_w2 = "", ""
max_dist = -inf
max_w1, max_w2 = "", ""

with all_ds.open("r") as f:
    f.readline()  # header
    for i, line in enumerate(f):
        w1, w2, d = line.split(",")
        d = float(d)
        if (w1 != w2) and (d < min_dist):
            min_dist = d
            min_w1, min_w2 = w1, w2
            print(i, w1, w2, d)
        elif d > max_dist:
            max_dist = d
            max_w1, max_w2 = w1, w2
            print(i, w1, w2, d)

print("---")
print(min_w1, min_w2, min_dist)
print(max_w1, max_w2, max_dist)
