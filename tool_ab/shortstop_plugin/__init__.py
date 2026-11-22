"""shortstop plug-in for mini-swe-agent: tool variants and a continue-or-stop hook."""

from .agent import EarlyStopped, StoppingAgent, ToolAgent
from .compress import CompressionConfig, compress_text
from .model import ToolModel
from .policies import constant, never_stop, tracking

__all__ = [
    "CompressionConfig",
    "EarlyStopped",
    "StoppingAgent",
    "ToolAgent",
    "ToolModel",
    "compress_text",
    "constant",
    "never_stop",
    "tracking",
]
