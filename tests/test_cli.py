import tempfile
import unittest
from pathlib import Path

from crossbar_analysis.cli import main


class CliTests(unittest.TestCase):
    def test_cli_writes_a_sweep(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "smoke.csv"
            status = main(
                [
                    "--device",
                    "mram",
                    "--dimension",
                    "16",
                    "--samples",
                    "64",
                    "--rs-points",
                    "5",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
