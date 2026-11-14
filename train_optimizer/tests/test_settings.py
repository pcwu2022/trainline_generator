from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train_optimizer.runner import main
from train_optimizer.settings import Config, DEFAULT_CONFIG_PATH, load_config, read_configuration


class SettingsTests(unittest.TestCase):
    def test_json_is_complete_and_is_the_only_source_of_defaults(self):
        defaults, profiles = read_configuration()
        self.assertEqual(set(defaults), {f.name for f in fields(Config)})
        self.assertEqual(asdict(Config()), defaults)
        self.assertEqual(set(profiles), {'europe', 'exploratory', 'france', 'italy', 'netherlands', 'switzerland', 'uk'})
        for name, overrides in profiles.items():
            with self.subTest(profile=name):
                config = load_config(profile=name)
                self.assertEqual(asdict(config), defaults | overrides)
                self.assertEqual(config.train_types, ['CR', 'HSR'])
                config.validate()

    def test_mutable_settings_are_independent_and_support_dataclass_replace(self):
        first, second = Config(), Config()
        first.train_types.clear()
        first.terrain_cost_ranges_million['hilly'][0] = 999
        self.assertEqual(second.train_types, ['CR', 'HSR'])
        self.assertEqual(second.terrain_cost_ranges_million['hilly'], [15, 25])
        self.assertEqual(replace(second, train_speed_kmh=150).train_speed_kmh, 150)
        self.assertEqual(second.train_speed_kmh, 120)

    def test_external_overrides_profile_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            external = Path(folder)/'external.json'
            external.write_text(json.dumps({'solver_time_limit': 123, 'ticket_per_km': .5}))
            config = load_config(external, 'italy')
            self.assertEqual(config.solver_time_limit, 600)
            self.assertEqual(config.ticket_per_km, .5)
            export = Path(folder)/'export.json'
            main(['--config', str(external), '--profile', 'italy', '--solver', 'milp',
                  '--write-default-config', str(export)])
            values = json.loads(export.read_text())
            self.assertEqual(values['solver_time_limit'], 600)
            self.assertEqual(values['solver_method'], 'milp')
            self.assertEqual(values['train_types'], ['CR', 'HSR'])
            self.assertEqual(asdict(load_config(export)), values)

    def test_canonical_file_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch('os.getcwd', return_value=folder):
            self.assertEqual(load_config(profile='uk').countries, ['GB'])
        self.assertTrue(DEFAULT_CONFIG_PATH.is_absolute())

    def test_unknown_profile_and_malformed_documents_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown profile'):
            load_config(profile='missing')
        for data in ([], {'defaults': []}, {'defaults': {}, 'profiles': {'bad': []}},
                     {'defaults': {}, 'extra': True}, {'unknown_setting': 1}):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'invalid.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_config(path)
