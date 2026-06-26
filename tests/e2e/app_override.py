"""
E2E application entrypoint.

This module is used *only* when running E2E tests inside Docker.
It patches src.core.llm.get_llm and src.core.llm.get_embedding_model
with the test-local mock implementations *before* the real
src.api.main module is imported, so the LangGraph workflow is
compiled with the mock LLM without any change to src/ code.

The Gunicorn command in docker-compose.e2e.yml points at:

    tests.e2e.app_override:app

instead of the usual:

    src.api.main:app

How patching works
------------------
Python's module system means that if we replace the function objects
in src.core.llm before src.core.workflow imports them, the workflow
will call our mocks.  We do the patch at module level here (before
any of those modules are imported) so the import order is:

  1. This file loads → patches src.core.llm
  2. src.api.main is imported → imports src.core.workflow
  3. src.core.workflow calls get_llm() / get_embedding_model()
     → gets mock instances

No monkeypatching library needed — a plain attribute assignment on
the already-imported module object is sufficient.
"""

# Step 1: import src.core.llm first so we hold a reference to the module
import src.core.llm as _llm_module

# Step 2: import the mock implementations from the E2E test folder
from tests.e2e.mock_llm import MockChatModel, MockEmbeddings


# Step 3: replace the factory functions in-place.
# Any subsequent import of get_llm / get_embedding_model from
# src.core.llm (or from src.core.workflow via its own import) will
# now resolve to these lambdas.
def _get_llm(model=None):  # type: ignore[override]
    return MockChatModel()


def _get_embedding_model(model=None):  # type: ignore[override]
    return MockEmbeddings()


_llm_module.get_llm = _get_llm  # type: ignore[assignment]
_llm_module.get_embedding_model = _get_embedding_model  # type: ignore[assignment]

# Step 4: now import the real FastAPI app — all its internal calls to
# get_llm() / get_embedding_model() will go through the mocks above.
from src.api.main import (  # noqa: E402, F401
    app,  # re-exported for Gunicorn
)

# Made with Bob
