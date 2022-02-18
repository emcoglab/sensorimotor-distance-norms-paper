Repository for code required to reproduce analysis for Wingfield & Connell's 2021 sensorimotor distance norms paper. Some were used in writing a response to reviews to a previous version of the manuscript.

Scripts:

- The main script, `main.py`: Computes distances between each pair of concepts used in the paper, using sensorimotor distances as well as the other predictors we compared and contrasted it with. Run this one if you want to reproduce the analysis in the paper exactly (though actual correlation and regression analyses are down downstream in [Jasp](https://jasp-stats.org) — see supplementary materials with the paper.)
- Some other scripts:
  - `closest_furthest.py`: Finds closest and furthest pairs.
  - `exclusivity_correlation.py`: Computes a correlation between the sensorimotor distance and the mean exclusivity ratings for randomly drawn pairs of concepts.
  - `perception_and_action_distance.py`: Reruns some of the analyses using both perception and action distance alongside sensorimotor distance.
