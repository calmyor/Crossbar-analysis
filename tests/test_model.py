import unittest

import numpy as np

from crossbar_analysis import (
    DEVICE_PRESETS,
    CrossbarConfig,
    Device,
    find_optimum,
    simulate_sweep,
)


class ModelTests(unittest.TestCase):
    def test_device_profiles_and_column_convention(self):
        self.assertAlmostEqual(DEVICE_PRESETS["mram"].contrast, 2.0)
        self.assertAlmostEqual(DEVICE_PRESETS["reram"].contrast, 12.0)
        self.assertAlmostEqual(DEVICE_PRESETS["fefet"].contrast, 1000.0)
        self.assertEqual(CrossbarConfig(dot_product_dimension=512).physical_columns, 1024)

    def test_sweep_is_deterministic_and_finite(self):
        config = CrossbarConfig(dot_product_dimension=32, samples=256, seed=9)
        grid = np.geomspace(100.0, 10_000.0, 9)
        first = simulate_sweep(DEVICE_PRESETS["reram"], config, grid)
        second = simulate_sweep(DEVICE_PRESETS["reram"], config, grid)
        np.testing.assert_allclose(
            [result.snr_db for result in first],
            [result.snr_db for result in second],
        )
        self.assertTrue(all(np.isfinite(result.snr_db) for result in first))
        self.assertIn(find_optimum(first), first)

    def test_invalid_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            CrossbarConfig(dot_product_dimension=0)
        with self.assertRaises(ValueError):
            Device("invalid", ron_ohm=10.0, roff_ohm=5.0)
        with self.assertRaises(ValueError):
            simulate_sweep(DEVICE_PRESETS["reram"], CrossbarConfig(samples=8), [])


if __name__ == "__main__":
    unittest.main()
