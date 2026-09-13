import unittest
from dataclasses import replace

import numpy as np

from crossbar_analysis.accuracy import (
    NOISE_CURVES,
    equivalent_weight_noise,
    predict_accuracy,
    snr_for_weight_noise,
)
from crossbar_analysis.model import DEVICE_PRESETS
from crossbar_analysis.parasitics import (
    ParasiticConfig,
    ideal_transfer,
    simulate_parasitic,
    transfer_matrix,
)


class ParasiticNetworkTests(unittest.TestCase):
    def setUp(self):
        self.device = DEVICE_PRESETS["reram"]
        rng = np.random.default_rng(7)
        self.conductance = rng.uniform(
            self.device.goff_siemens, self.device.gon_siemens, size=(3, 16)
        )
        self.ideal_wires = ParasiticConfig(
            dot_product_dimension=8,
            active_rows=3,
            r_bl_ohm=0.0,
            r_sl_ohm=0.0,
            r_driver_ohm=0.0,
            r_sense_ohm=500.0,
        )

    def test_ideal_wires_match_closed_form_current_sensing(self):
        solved = transfer_matrix(self.conductance, self.ideal_wires)
        closed = ideal_transfer(self.conductance, self.ideal_wires)
        np.testing.assert_allclose(solved, closed, rtol=1e-5)

    def test_ideal_wires_match_closed_form_voltage_sensing(self):
        config = replace(self.ideal_wires, sensing="voltage")
        solved = transfer_matrix(self.conductance, config)
        closed = ideal_transfer(self.conductance, config)
        np.testing.assert_allclose(solved, closed, rtol=1e-5)

    def test_wire_resistance_compresses_gain_as_dimension_grows(self):
        gains = []
        for dimension in (16, 64, 256):
            config = ParasiticConfig(
                dot_product_dimension=dimension,
                conductance_variation_ratio=0.0,
                dac_mismatch_ratio=0.0,
                r_sense_ohm=100.0,
                weight_trials=1,
            )
            gains.append(simulate_parasitic(DEVICE_PRESETS["mram"], config).mean_gain)
        self.assertGreater(gains[0], gains[1])
        self.assertGreater(gains[1], gains[2])

    def test_pure_ir_drop_is_removed_by_design_aware_mapping(self):
        config = ParasiticConfig(
            dot_product_dimension=32,
            active_rows=4,
            conductance_variation_ratio=0.0,
            dac_mismatch_ratio=0.0,
            r_sense_ohm=100.0,
            weight_trials=1,
        )
        result = simulate_parasitic(DEVICE_PRESETS["mram"], config)
        self.assertLess(result.raw.analog_snr_db, 40.0)
        self.assertGreater(result.ir_aware.analog_snr_db, 100.0)

    def test_nonlinearity_limits_chip_in_the_loop_compensation(self):
        base = ParasiticConfig(
            dot_product_dimension=32,
            active_rows=4,
            conductance_variation_ratio=0.0,
            dac_mismatch_ratio=0.0,
            r_sense_ohm=100.0,
            weight_trials=1,
        )
        linear = simulate_parasitic(self.device, base)
        nonlinear = simulate_parasitic(self.device, replace(base, iv_nonlinearity=0.05))
        self.assertGreater(linear.chip_in_loop.analog_snr_db, 100.0)
        self.assertLess(nonlinear.chip_in_loop.analog_snr_db, 60.0)

    def test_input_rail_sag_is_not_removed_by_calibration(self):
        base = ParasiticConfig(
            dot_product_dimension=32,
            active_rows=8,
            conductance_variation_ratio=0.0,
            dac_mismatch_ratio=0.0,
            r_sense_ohm=100.0,
            weight_trials=1,
        )
        clean = simulate_parasitic(self.device, base)
        sagging = simulate_parasitic(
            self.device, replace(base, r_input_rail_ohm=5.0, parallel_cores=4)
        )
        self.assertLess(
            sagging.calibrated.analog_snr_db, clean.calibrated.analog_snr_db - 10.0
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            ParasiticConfig(sensing="charge")
        with self.assertRaises(ValueError):
            ParasiticConfig(dac_inl_lsb=-0.1)


class AccuracyMappingTests(unittest.TestCase):
    def test_weight_noise_round_trip(self):
        for noise in (0.02, 0.1, 0.2):
            self.assertAlmostEqual(equivalent_weight_noise(snr_for_weight_noise(noise)), noise)

    def test_accuracy_falls_with_snr_and_stays_bounded(self):
        curve = NOISE_CURVES["resnet20-cifar10-injection-20"]
        high = predict_accuracy(40.0, curve)
        low = predict_accuracy(8.0, curve)
        self.assertGreaterEqual(high.accuracy, low.accuracy)
        self.assertLessEqual(high.accuracy, curve.clean_accuracy)
        self.assertGreaterEqual(low.accuracy, curve.chance)
        self.assertTrue(low.extrapolated)
        self.assertEqual(low.accuracy, curve.points[-1][1])

    def test_noise_injection_training_is_more_tolerant(self):
        snr = snr_for_weight_noise(0.05)
        trained = predict_accuracy(snr, NOISE_CURVES["resnet20-cifar10-injection-20"])
        untrained = predict_accuracy(snr, NOISE_CURVES["resnet20-cifar10-no-injection"])
        self.assertGreater(trained.retention, untrained.retention)


if __name__ == "__main__":
    unittest.main()
