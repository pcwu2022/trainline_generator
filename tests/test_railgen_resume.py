"""Checkpoint continuation and compatibility checks."""
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from test_railgen_live import incremental_config
from test_railgen import graph_fixture
from rail_gen.checkpoint import load_checkpoint
from rail_gen.search import search
from rail_gen.track_layout import MODEL_VERSION


class ResumeTests(unittest.TestCase):
    def test_checkpoint_restores_network_and_numbering(self):
        c, g = incremental_config(), graph_fixture()
        updates = []
        search(g, c, progress=None, observer=updates.append)
        update = next(u for u in updates if u.accepted)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.npz'
            np.savez(path, genome=update.result.genome, track_counts=update.result.track_counts,
                     iteration=update.iteration, graph_fingerprint=g.metadata.get('fingerprint', ''),
                     config_json=json.dumps(c), model_version=MODEL_VERSION,
                     services_json=json.dumps(update.result.services),
                     search_state_json=json.dumps(update.metadata['search_state']))
            c['search']['generations'] = 2
            saved = load_checkpoint(path, g, c)
            resumed = []
            search(g, c, progress=None, observer=resumed.append, resume=saved)
            self.assertEqual(resumed[0].iteration, update.iteration)
            self.assertEqual(resumed[0].phase, 'resumed')
            np.testing.assert_array_equal(resumed[0].result.genome, update.result.genome)
            np.testing.assert_array_equal(resumed[0].result.track_counts, update.result.track_counts)
            if len(resumed) > 1:
                self.assertEqual(resumed[1].iteration, update.iteration + 1)
                np.testing.assert_array_equal(resumed[1].parent_genome, update.result.genome)
            c['demand']['k'] *= 2
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_checkpoint(path, g, c)

    def test_population_resume_rejected(self):
        c = incremental_config()
        c['search']['algorithm'] = 'nsga2'
        with self.assertRaisesRegex(ValueError, 'incremental'):
            search(graph_fixture(), c, progress=None, resume={})
