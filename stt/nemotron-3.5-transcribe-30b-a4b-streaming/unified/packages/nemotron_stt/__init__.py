"""Nemotron 3.5 Transcribe behind baseten-stt's backend interface.

Lives under ``packages/`` because truss copies that directory to ``/packages``, which
``PYTHONPATH`` puts on ``sys.path`` -- so ``BASETEN_STT_BACKEND=nemotron_stt:build`` resolves.
"""

from .backend import DEFAULT_PARTIAL_WINDOW_S, NemotronBackend, NemotronStream
from .engine import build

#: ``build`` is re-exported here rather than reached at ``nemotron_stt.engine:build``, so the
#: deployment names the package and not the file the load happens to live in today.
__all__ = ["DEFAULT_PARTIAL_WINDOW_S", "NemotronBackend", "NemotronStream", "build"]
