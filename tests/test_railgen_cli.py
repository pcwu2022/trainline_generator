"""Output replacement safeguards for both fast-generator entry points."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rail_gen.__main__ import main
from train_optimizer.runner import main as optimizer_main


class RailgenCliTests(unittest.TestCase):
    def test_overwrite_clears_output_through_both_entry_points(self):
        for entry, prefix in [(main, ['run']), (optimizer_main, ['--solver', 'pareto'])]:
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'run'
                (output / 'snapshots').mkdir(parents=True)
                (output / 'snapshots' / 'old.json').write_text('old')

                def stop_before_preparation(*args):
                    self.assertFalse(output.exists())
                    raise RuntimeError('stop before preparation')

                with patch('rail_gen.alignment.cached_graph', side_effect=stop_before_preparation):
                    with self.assertRaisesRegex(RuntimeError, 'stop before preparation'):
                        entry([*prefix, '--output', str(output), '--overwrite'])

    def test_rejected_options_preserve_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            output.mkdir()
            marker = output / 'keep.txt'
            marker.write_text('keep')
            link = Path(folder) / 'link'
            link.symlink_to(output, target_is_directory=True)
            for options in [
                ['--output', str(output)],
                ['--output', str(output), '--overwrite', '--snapshot-every', '0'],
                ['--output', str(link), '--overwrite'],
                ['--output', str(marker), '--overwrite'],
                ['--output', '.', '--overwrite'],
                ['--output', str(Path.cwd().parent), '--overwrite'],
            ]:
                with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                    with patch('rail_gen.alignment.cached_graph') as prepare:
                        with self.assertRaises(SystemExit):
                            main(['run', *options])
                        prepare.assert_not_called()
                    self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
