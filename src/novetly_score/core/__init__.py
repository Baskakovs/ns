"""The novelty score as the paper defines it: one module per section of `Novelty Paper.md`.

Scalar, pure, and free of I/O. This is the readable definition — the thing to check
line-by-line against the paper — and `..engine` holds the vectorised implementation that
actually runs, with an equivalence test pinning the two together.

**Nothing here may import from `..engine`.** The dependency runs one way, so that the
definition cannot come to depend on the optimisation.
"""
