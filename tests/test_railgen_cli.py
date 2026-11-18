"""Output replacement safeguards for the generator CLI."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rail_gen.__main__ import main


class RailgenCliTests(unittest.TestCase):
    def test_overwrite_clears_output(self):
        for entry, prefix in [(main, ['run'])]:
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

    def test_city_limit_overrides_profile_before_graph_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            for command in ('precompute', 'run', 'benchmark'):
                for count in (50, 100):
                    with self.subTest(command=command, count=count):
                        args=[command, '--profile', 'india', '--max-cities', str(count)]
                        if command=='run':
                            args.extend(['--output', str(Path(folder)/'output')])
                        with patch('rail_gen.alignment.cached_graph', side_effect=RuntimeError('preparation reached')) as prepare:
                            with self.assertRaisesRegex(RuntimeError,'preparation reached'):
                                main(args)
                            self.assertEqual(prepare.call_args.args[0]['prep']['max_cities'],count)

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
                ['--output', str(output), '--overwrite', '--max-cities', '1'],
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
