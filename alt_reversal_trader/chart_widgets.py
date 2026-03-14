from __future__ import annotations

from importlib import import_module
from typing import Any


QtChart: Any = import_module("lightweight_charts.widgets").QtChart
