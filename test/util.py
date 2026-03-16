import unittest
from pathlib import Path
import pandas as pd

from lightweight_charts import Chart


_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "1_setting_data"
BARS = pd.read_csv(_EXAMPLES_DIR / "ohlcv.csv")



class Tester(unittest.TestCase):
    def setUp(self):
        self.chart: Chart = Chart(100, 100, 800, 100);

    def tearDown(self) -> None:
        self.chart.exit()



