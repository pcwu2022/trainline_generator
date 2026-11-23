"""Renamed uploads retain existing video continuation data and report links."""
import tempfile
from pathlib import Path
import unittest
from rail_gen.output_names import migrate_visual_names, visual_names


class VisualNamesTests(unittest.TestCase):
    def test_country_name_and_upload_order(self):
        names = visual_names({'reporting': {'region_name': 'United Kingdom'}})
        self.assertEqual(list(names.values()), [
            '_united_kingdom_1_animation.mp4', '_united_kingdom_2_analysis1.png',
            '_united_kingdom_3_analysis2.png', '_united_kingdom_4_progress.png',
            '_united_kingdom_5_map.png'])

    def test_migration_preserves_video_state_and_updates_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'progress.mp4').write_bytes(b'video')
            (output / 'progress.json').write_text('{"snapshots": 9, "size": [1080,1080]}')
            (output / 'report.md').write_text('[Video](progress.mp4)\n![Map](latest.png)\n![Costs](latest_revenue_cost.png)\n')
            c = {'reporting': {'region_name': 'Germany'}}
            names = migrate_visual_names(output, c)
            self.assertEqual((output / names['animation']).read_bytes(), b'video')
            state = output / 'progress.json'
            self.assertIn('"snapshots": 9', state.read_text())
            self.assertFalse((output / '_germany_1_animation.json').exists())
            self.assertIn('(_germany_5_map.png)', (output / 'report.md').read_text())
            self.assertNotIn('latest_revenue_cost', (output / 'report.md').read_text())
            migrate_visual_names(output, c)
            self.assertEqual((output / names['animation']).read_bytes(), b'video')

    def test_previously_renamed_json_is_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / '_germany_1_animation.json').write_text('{"snapshots": 3}')
            migrate_visual_names(output, {'reporting': {'region_name': 'Germany'}})
            self.assertEqual((output / 'progress.json').read_text(), '{"snapshots": 3}')
            self.assertFalse((output / '_germany_1_animation.json').exists())
