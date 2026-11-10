import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train_optimizer.runner import main


class RunnerCliTests(unittest.TestCase):
    def test_overwrite_clears_old_output_before_reporter(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            (output / 'snapshots').mkdir(parents=True)
            (output / 'snapshots' / 'old.json').write_text('old')

            def reporter(*args):
                self.assertFalse(output.exists())
                raise RuntimeError('stop before running solver')

            with patch('train_optimizer.runner.Reporter', side_effect=reporter):
                with self.assertRaisesRegex(RuntimeError, 'stop before running solver'):
                    main(['--output', str(output), '--overwrite'])

    def test_rejected_options_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / 'keep.txt'
            marker.write_text('keep')
            for options in ([], ['--overwrite', '--resume', str(output / 'checkpoint.npz')]):
                with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(['--output', str(output), *options])
                    self.assertEqual(error.exception.code, 2)
                    self.assertEqual(marker.read_text(), 'keep')

    def test_overwrite_rejects_working_directory(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--output', '.', '--overwrite'])
            self.assertEqual(error.exception.code, 2)
