import unittest
import pandas as pd
from util import BARS, Tester
from lightweight_charts import Chart
from lightweight_charts import abstract


class TestChart(Tester):
    def test_data_is_renamed(self):
        uppercase_df = pd.DataFrame(BARS.copy()).rename({'date': 'Date', 'open': 'OPEN', 'high': 'HIgh', 'low': 'Low', 'close': 'close', 'volUME': 'volume'})
        result = self.chart._df_datetime_format(uppercase_df)
        self.assertEqual(list(result.columns), list(BARS.rename(columns={'date': 'time'}).columns))

    def test_line_in_list(self):
        result0 = self.chart.create_line()
        result1 = self.chart.create_line()
        self.assertEqual(result0, self.chart.lines()[0])
        self.assertEqual(result1, self.chart.lines()[1])

    def test_scale_candles_only_line_js_has_comma(self):
        win = abstract.Window(script_func=lambda _: None)
        chart = abstract.AbstractChart(win, scale_candles_only=True)
        chart.create_line("Equity")
        script = win.scripts[-1]
        self.assertIn('priceScaleId: undefined,', script)
        self.assertIn('autoscaleInfoProvider', script)

    def test_datetime_format_uses_epoch_seconds(self):
        result = self.chart._df_datetime_format(BARS.copy())
        self.assertGreater(result["time"].iloc[0], 1_000_000_000)
        self.assertTrue(result["time"].is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
