"""The satisfaction-driven model replaces the optional profit switch."""
import contextlib
import io
import unittest

from rail_gen.__main__ import parser
from rail_gen.config import load_config


class ProfitRemovalTests(unittest.TestCase):
    def test_profit_cli_removed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['run', '--profit'])

    def test_profit_config_removed(self):
        with self.assertRaisesRegex(ValueError, 'Unknown parameter: search.profit'):
            load_config(overrides={'search': {'profit': True}})
