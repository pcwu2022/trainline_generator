"""Prepared browser scenarios run the real optimizer without the input raster."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class WebRunnerTests(unittest.TestCase):
    def run_scenario(self, count, mutate=None, profile='france'):
        messages = []
        bridge = SimpleNamespace(postMessage=lambda message: messages.append(json.loads(message)))
        previous = sys.modules.get('js')
        sys.modules['js'] = bridge
        try:
            run = runpy.run_path(str(ROOT/'web/runner.py'))['run_browser']
        finally:
            if previous is None:
                sys.modules.pop('js', None)
            else:
                sys.modules['js'] = previous
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(ROOT/f'web/scenarios/{profile}/cities-{count}/graph.zip') as archive:
                archive.extractall(directory)
            config=json.loads((Path(directory)/'graph.json').read_text())['config']
            config['search']['generations']=1
            if mutate:
                mutate(config)
            with redirect_stdout(io.StringIO()):
                run(json.dumps(config), directory)
        return messages

    def test_every_city_choice_streams_matching_graph(self):
        for count in (4,8,12,20,50,100):
            with self.subTest(cities=count):
                display=json.loads((ROOT/f'web/scenarios/france/cities-{count}/scenario.json').read_text())
                actual=len(display['cities'])
                messages=self.run_scenario(count)
                updates=[m for m in messages if m['type']=='update']
                self.assertEqual([u['iteration'] for u in updates],[0,1])
                for update in updates:
                    self.assertEqual(sum(f['geometry']['type']=='Point' for f in update['network']['features']),actual)
                self.assertEqual(messages[-1]['type'],'done')
                self.assertEqual(messages[-1]['config']['prep']['max_cities'],count)

    def test_zero_demand_setting_reaches_optimizer(self):
        messages=self.run_scenario(4,lambda c:c['demand'].update(k=0))
        self.assertEqual(messages[-1]['config']['demand']['k'],0)
        self.assertEqual([m for m in messages if m['type']=='update'][-1]['metrics']['passengers'],0)

    def test_mismatched_city_count_requires_matching_bundle(self):
        with self.assertRaisesRegex(ValueError,'prep is baked'):
            self.run_scenario(4,lambda c:c['prep'].update(max_cities=8))

    def test_profile_catalog_and_default_graphs_match_their_region(self):
        catalog=json.loads((ROOT/'web/scenarios/index.json').read_text())
        expected={name:None for name in ('france','taiwan','usa','uk','japan','germany','spain','italy','poland','benelux','switzerland','india','scandinavia','usa_northeast')}
        self.assertTrue(set(expected).issubset({p['id'] for p in catalog['profiles']}))
        for profile in catalog['profiles']:
            with self.subTest(profile=profile['id']):
                folder=ROOT/'web/scenarios'/profile['path']
                index=json.loads((folder/'index.json').read_text())
                self.assertEqual(index['profile'],profile['id'])
                if profile['id'] in expected:
                    self.assertEqual([c['city_count'] for c in index['scenarios']],[4,8,12,20,50,100])
                from rail_gen.config import load_config
                countries=load_config(profile=profile['id'])['countries']
                for choice in index['scenarios']:
                    scenario=folder/choice['path']
                    self.assertTrue((scenario/'background.png').is_file())
                    display=json.loads((scenario/'scenario.json').read_text())
                    with zipfile.ZipFile(scenario/'graph.zip') as archive:
                        descriptor=json.loads(archive.read('graph.json'))
                    self.assertEqual(descriptor['config'],display['config'])
                    self.assertEqual(descriptor['config']['countries'],countries)
                    self.assertEqual(len(descriptor['cities']),choice['actual_city_count'])
                    self.assertLessEqual(choice['actual_city_count'],choice['city_count'])
                    self.assertGreaterEqual(choice['actual_city_count'],2)
                messages=self.run_scenario(index['default_city_count'],profile=profile['id'])
                updates=[m for m in messages if m['type']=='update']
                self.assertEqual([m['iteration'] for m in updates],[0,1])
                self.assertEqual(messages[-1]['config']['countries'],countries)
