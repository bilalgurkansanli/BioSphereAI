"""HTTP layer.

Deliberately empty of imports: ``biosphere.core.jobs`` imports ``biosphere.api.schemas`` for
the event payload converters, so pulling ``main`` in here would create an import cycle
(``api`` -> ``main`` -> ``routes`` -> ``core.jobs`` -> ``api.schemas``). Import
:func:`biosphere.api.main.create_app` from its own module instead.
"""

from __future__ import annotations

__all__: list[str] = []
