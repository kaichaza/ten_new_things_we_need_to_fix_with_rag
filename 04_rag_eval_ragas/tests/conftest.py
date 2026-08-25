import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# ragas cannot be imported until the shim is installed; see
# _compat.py. ROOT is already on sys.path above.
import _compat  # noqa: E402,F401
