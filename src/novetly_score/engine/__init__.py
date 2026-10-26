"""Vectorised implementation and I/O for the novelty score.

Everything here exists for scale. The readable definition of what these modules compute
lives in ``..core``, and an equivalence test pins the two together. ``core`` must never
import from here.

**Importing this package checks that pinning is current.** If a core definition or its engine
implementation changed since their equivalence tests last passed, the import raises
:class:`honey.novelty_score_exp.twins.CoreDivergenceError` — so the engine cannot run a definition
`core/` no longer holds. See :mod:`honey.novelty_score_exp.twins`.
"""

from honey.novelty_score_exp.twins import verify

verify()
