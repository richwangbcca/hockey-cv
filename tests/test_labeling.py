"""Test labeling dataset persistence, validation, geometry, and export behavior."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from labeling.app import (Dataset, LANDMARKS, atomic_json, export_dataset, fit_geometry,
                          import_legacy, index_images, split_groups, validate_record,
                          validate_dataset, sample_timed_frames, migrate_binary_usability)


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        cv2.imwrite(str(self.images / '001.png'), np.full((600, 1000, 3), 190, np.uint8))
        self.path = self.root / 'dataset'
        index_images(self.images, self.path, 'test')
        self.dataset = Dataset(self.path)
        self.key = next(iter(self.dataset.entries))

    def tearDown(self):
        self.temp.cleanup()

    def observations(self):
        record = self.dataset.get(self.key)
        names = ['dotNZ_W_N', 'dotNZ_E_N', 'dotNZ_W_S', 'dotNZ_E_S', 'center']
        record['features'] = [dict(id=name, type='landmark', landmark=name,
            points=[[500 + LANDMARKS[name][0]*5, 300 - LANDMARKS[name][1]*5]],
            visibility='visible', provenance='manual', held_out=name == 'center') for name in names]
        record['orientation'] = 'sequence_local'
        return record

    def test_reindex_preserves_annotations_and_unknown_timing(self):
        record = self.dataset.get(self.key)
        record['notes'] = 'keep me'
        self.dataset.save(record)
        index_images(self.images, self.path, 'test')
        self.assertEqual(Dataset(self.path).get(self.key)['notes'], 'keep me')
        self.assertIsNone(self.dataset.entries[self.key]['timestamp_seconds'])

    def test_source_mutation_is_rejected(self):
        cv2.imwrite(str(self.images / '001.png'), np.zeros((600, 1000, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            index_images(self.images, self.path, 'test')

    def test_revision_conflict_does_not_overwrite(self):
        a = self.dataset.get(self.key)
        b = copy.deepcopy(a)
        a['notes'] = 'first edit'
        self.dataset.save(a)
        with self.assertRaisesRegex(ValueError, 'Revision conflict'):
            self.dataset.save(b)
        self.assertEqual(self.dataset.get(self.key)['notes'], 'first edit')

    def test_fit_holdout_and_edit_invalidation(self):
        self.dataset.save(self.observations())
        record = self.dataset.fit(self.key)
        self.assertEqual(record['calibration']['validation_count'], 1)
        self.assertLess(record['calibration']['validation'][0]['error_px'], .001)
        record['shot']['geometry_status'] = 'accepted'
        record = self.dataset.save(record)
        record['features'][0]['points'][0][0] += 3
        record = self.dataset.save(record)
        self.assertIsNone(record['calibration'])
        self.assertEqual(record['shot']['geometry_status'], 'unreviewed')

    def test_unseen_coordinates_and_hidden_contact_rejected(self):
        record = self.observations()
        record['features'][0]['visibility'] = 'occluded'
        with self.assertRaisesRegex(ValueError, 'Unseen'):
            validate_record(record, self.dataset.entries[self.key])
        record['features'] = []
        record['players'] = [dict(id='1', role='skater', bbox=[20, 20, 50, 80],
            ice_point=[35, 80], contact_visibility='hidden', provenance='manual', occluded=True, truncated=False)]
        with self.assertRaisesRegex(ValueError, 'Hidden feet'):
            validate_record(record, self.dataset.entries[self.key])

    def test_player_team_roundtrip_and_export(self):
        record = self.dataset.get(self.key)
        record['players'] = [dict(id='p1', role='skater', team='team_a',
            bbox=[20, 20, 50, 80], ice_point=[35, 80], contact_visibility='both',
            provenance='manual', occluded=False, truncated=False)]
        record['review']['players'] = 'reviewed'
        self.dataset.save(record)
        record = self.dataset.get(self.key)
        self.assertEqual(record['players'][0]['team'], 'team_a')
        destination = self.root / 'export'
        export_dataset(self.dataset, destination)
        row = json.loads((destination / 'players.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['players'][0]['team'], 'team_a')
        record['players'][0]['team'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'Invalid team'):
            self.dataset.save(record)

    def test_acceptance_allows_four_points_without_holdout(self):
        record = self.observations()
        record['features'] = record['features'][:4]
        self.dataset.save(record)
        record = self.dataset.fit(self.key)
        record['shot']['geometry_status'] = 'accepted'
        record = self.dataset.save(record)
        self.assertEqual(self.dataset.get(self.key)['shot']['geometry_status'], 'accepted')
        record['review']['geometry'] = 'reviewed'
        record = self.dataset.save(record)
        self.assertEqual(record['review']['geometry'], 'reviewed')
        self.assertEqual(record['calibration']['validation_count'], 0)
        missing_fit = copy.deepcopy(record)
        missing_fit['calibration'] = None
        with self.assertRaisesRegex(ValueError, 'calculated fit'):
            validate_record(missing_fit, self.dataset.entries[self.key])

    def test_duplicate_and_collinear_anchors_rejected(self):
        record = self.observations()
        record['features'][1]['landmark'] = record['features'][0]['landmark']
        with self.assertRaisesRegex(ValueError, 'once'):
            fit_geometry(record)
        record = self.observations()
        for i, feature in enumerate(record['features']):
            feature['points'] = [[100+i*10, 100]]
        with self.assertRaisesRegex(ValueError, 'collinear'):
            fit_geometry(record)

    def test_legacy_import_requires_matching_bytes_and_stays_proposal(self):
        labels = self.root / 'labels'
        labels.mkdir()
        atomic_json(labels / '001.png.json', {'center': [500, 300]})
        import_legacy(self.dataset, labels, self.images)
        import_legacy(self.dataset, labels, self.images)
        record = self.dataset.get(self.key)
        self.assertEqual(len(record['proposals']), 1)
        self.assertEqual(record['features'], [])
        self.assertEqual(record['review']['geometry'], 'unreviewed')
        other = self.root / 'other'
        other.mkdir()
        cv2.imwrite(str(other / '001.png'), np.zeros((600, 1000, 3), np.uint8))
        import_legacy(self.dataset, labels, other)
        self.assertEqual(len(self.dataset.get(self.key)['proposals']), 1)

    def test_export_omits_unreviewed_and_limits_derived_points(self):
        out = self.root / 'export'
        export_dataset(self.dataset, out)
        self.assertEqual((out / 'shots.jsonl').read_text(), '')
        record = self.observations()
        record['players'] = [dict(id=str(i), role='skater', bbox=[10, 10, 30, 60],
            ice_point=p, contact_visibility=visibility, provenance='manual', occluded=False, truncated=False)
            for i, (p, visibility) in enumerate([([500, 300], 'both'), ([20, 20], 'both'), ([500, 300], 'estimated')])]
        self.dataset.save(record)
        record = self.dataset.fit(self.key)
        record['shot']['geometry_status'] = 'accepted'
        record['shot']['position_usability'] = 'usable'
        record['review'] = dict(shot='reviewed', geometry='reviewed', players='reviewed')
        self.dataset.save(record)
        export_dataset(self.dataset, out)
        players = json.loads((out / 'players.jsonl').read_text())['players']
        self.assertTrue(np.allclose(players[0]['rink_point_derived'], [0, 0]))
        self.assertIsNone(players[1]['rink_point_derived'])
        self.assertIsNone(players[2]['rink_point_derived'])
        self.assertFalse(validate_dataset(self.dataset)['errors'])

    def test_duplicate_images_union_groups(self):
        (self.images / '002.png').write_bytes((self.images / '001.png').read_bytes())
        index_images(self.images, self.path, 'test')
        ds = Dataset(self.path)
        self.assertEqual(set(split_groups(ds).values()), {'unassigned'})
        for i, key in enumerate(ds.entries):
            record = ds.get(key)
            record['shot_id'] = 'shot'+str(i)
            ds.save(record)
        self.assertEqual(len(set(split_groups(ds).values())), 1)

    def test_sampling_requires_timing_and_does_not_fill_gaps(self):
        out = self.root / 'samples.json'
        sample_timed_frames(self.dataset, out, [2, 4])
        self.assertEqual(json.loads(out.read_text())['selections'], {'2': [], '4': []})
        times = [0, .25, .5, 1.0]
        for i in range(1, 4):
            cv2.imwrite(str(self.images / f'{i+1:03}.png'), np.full((600, 1000, 3), 190+i, np.uint8))
        timing = self.root / 'timing.json'
        atomic_json(timing, {f'{i+1:03}.png': {'timestamp_seconds': t} for i, t in enumerate(times)})
        index_images(self.images, self.path, 'test', timing)
        ds = Dataset(self.path)
        for key in ds.entries:
            record = ds.get(key); record['shot_id'] = 'continuous'; ds.save(record)
        sample_timed_frames(ds, out, [2, 4])
        result = json.loads(out.read_text())['selections']
        self.assertEqual([r['timestamp_seconds'] for r in result['2']], [0, .5, 1.0])
        self.assertEqual([r['timestamp_seconds'] for r in result['4']], times)

    def test_completed_first_phase_requires_binary_usability(self):
        for usability in ('unknown', 'partial'):
            with self.subTest(usability=usability):
                record = self.dataset.get(self.key)
                record['shot']['position_usability'] = usability
                record['review']['shot'] = 'reviewed'
                with self.assertRaisesRegex(ValueError, 'usable or unusable'):
                    self.dataset.save(record)
        for usability in ('usable', 'unusable'):
            record = self.dataset.get(self.key)
            record['shot']['position_usability'] = usability
            record['review']['shot'] = 'reviewed'
            self.dataset.save(record)

    def test_stale_fit_request_rejected(self):
        record = self.dataset.save(self.observations())
        with self.assertRaisesRegex(ValueError, 'Revision conflict'):
            self.dataset.fit(self.key, record['revision'] - 1)

    def test_landmarks_finish_before_calibration_and_export(self):
        record = self.dataset.save(self.observations())
        record['review']['landmarks'] = 'reviewed'
        self.dataset.save(record)
        out = self.root / 'export'
        export_dataset(self.dataset, out)
        self.assertTrue((out / 'features.jsonl').read_text())
        self.assertEqual((out / 'calibrations.jsonl').read_text(), '')
        record = self.dataset.get(self.key)
        record['features'][0]['held_out'] = True
        record = self.dataset.save(record)
        self.assertEqual(record['review']['landmarks'], 'reviewed')
        record['features'][0]['points'][0][0] += 1
        record = self.dataset.save(record)
        self.assertEqual(record['review']['landmarks'], 'in_progress')

    def test_legacy_review_upgrade_preserves_saved_record(self):
        record = self.dataset.get(self.key)
        record['review'].pop('landmarks')
        record['review']['geometry'] = 'reviewed'
        path = self.path / 'records' / (self.key + '.json')
        atomic_json(path, record)
        before = path.read_bytes()
        self.assertEqual(self.dataset.get(self.key)['review']['landmarks'], 'reviewed')
        self.assertEqual(path.read_bytes(), before)

    def test_partial_migration_preserves_history_and_review(self):
        record = self.dataset.get(self.key)
        record['shot']['position_usability'] = 'partial'
        record['review']['shot'] = 'unreviewed'
        self.dataset.save(record)
        # Simulate a legacy completed record, which the new validator will no
        # longer save until migration changes the value.
        record = self.dataset.get(self.key)
        record['review']['shot'] = 'reviewed'
        atomic_json(self.path / 'records' / (self.key + '.json'), record)
        migrate_binary_usability(self.dataset)
        migrated = self.dataset.get(self.key)
        self.assertEqual(migrated['shot']['position_usability'], 'usable')
        self.assertEqual(migrated['review']['shot'], 'reviewed')
        self.assertEqual(migrated['migration_history'][-1]['previous_value'], 'partial')


if __name__ == '__main__':
    unittest.main()
