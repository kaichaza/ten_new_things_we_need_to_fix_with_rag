"""Compatibility shim: restore langchain_community.chat_models.vertexai.

WHY THIS EXISTS

ragas executes this line at module load, in every version from 0.3 through
at least 0.4.3, in ragas/llms/base.py:

    from langchain_community.chat_models.vertexai import ChatVertexAI

ChatVertexAI was deprecated in langchain-community 0.0.12 and moved to the
separate langchain-google-vertexai package. During langchain-community's
sunset the old module was removed outright, so that import now raises
ModuleNotFoundError - and because it runs at import time, "import ragas"
fails before any user code runs, for everyone not on Google Cloud.

Upstream issues: ragas 2741, 2745, 2753.

WHY A SHIM RATHER THAN A PIN

Pinning ragas does not help: every candidate version makes the same import.
Pinning langchain-community would work, but requires knowing which release
last shipped the module - an answer that goes stale. The shim works
regardless of which versions resolve.

WHAT IT DOES

Registers a module at the missing path exposing a ChatVertexAI name, so the
import succeeds. Nothing in this exercise uses VertexAI; the class raises if
anyone actually tries to construct it, so the shim cannot quietly become a
broken code path.

Delete this file and its import in main.py once ragas ships the fix.
"""
import sys
import types


def install() -> bool:
    """Register the shim if the real module is missing. Returns True if used."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        """Placeholder. Present so ragas can import it, never constructed."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI is a compatibility placeholder in this exercise. "
                "VertexAI is not used here; see _compat.py."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module

    # Also attach to the parent package, so attribute-style access resolves.
    parent = sys.modules.get("langchain_community.chat_models")
    if parent is not None:
        setattr(parent, "vertexai", module)
    return True


install()
