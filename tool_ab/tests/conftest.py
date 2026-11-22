import sys
from pathlib import Path

# The plug-in and the experiment scripts live in tool_ab/, which is not an installed package.
TOOL_AB = Path(__file__).resolve().parents[1]
if str(TOOL_AB) not in sys.path:
    sys.path.insert(0, str(TOOL_AB))
