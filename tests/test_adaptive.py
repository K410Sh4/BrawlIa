"""Testes puros e determinísticos para o controlador de desempenho."""

import unittest

from brawlla.adaptive import AdaptiveFps, percentile, stepped_limit


class AdaptiveTests(unittest.TestCase):
    def test_percentile_and_steps(self):
        self.assertEqual(percentile([], 0.9), 0.0)
        self.assertEqual(percentile(range(1, 11), 0.9), 9)
        self.assertEqual(stepped_limit(27), 24)
        self.assertEqual(stepped_limit(1), 5)

    def _feed(self, controller, ms, cap_ms, start, end, age=None):
        for i in range(start, end):
            controller.record_capture(cap_ms)
            controller.record_analysis(ms, ms if age is None else age, now=i * 0.2)

    def test_slow_analysis_reduces_rate_and_capture_without_lag_queue(self):
        ctl = AdaptiveFps(adjust_interval_s=0.5)
        self._feed(ctl, 125, 20, 0, 20)
        cap, ana = ctl.rates()
        self.assertLess(ana, 15)
        self.assertLess(cap, 30)
        self.assertGreaterEqual(cap, ana)
        self.assertGreater(ctl.snapshot()["adjustments"], 0)

    def test_fast_machine_can_ramp_up_but_not_above_sixty(self):
        ctl = AdaptiveFps(adjust_interval_s=0.5)
        self._feed(ctl, 5, 3, 0, 100)
        cap, ana = ctl.rates()
        self.assertGreater(ana, 15)
        self.assertLessEqual(ana, 60)
        self.assertLessEqual(cap, 60)

    def test_backlog_age_blocks_rate_increase(self):
        ctl = AdaptiveFps(adjust_interval_s=0.5)
        self._feed(ctl, 8, 5, 0, 40, age=250)
        self.assertEqual(ctl.rates()[1], 15)
        self.assertEqual(ctl.snapshot()["reason"], "fila_atrasada")

    def test_reject_invalid_data_and_parameters(self):
        with self.assertRaises(ValueError):
            AdaptiveFps(capture_fps=11)
        ctl = AdaptiveFps()
        ctl.record_capture(float("nan"))
        ctl.record_analysis(float("nan"), 0, now=0)
        self.assertEqual(ctl.snapshot()["capture_p90_ms"], 0.0)
        self.assertEqual(ctl.snapshot()["analysis_p90_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
