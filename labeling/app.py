"""Run the offline annotation server and manage hockey-image labeling datasets."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
from vision.rink import LANDMARKS

VERSION = 1
TEMPLATE = 'bench-center-feet-v1-provisional'
TAGS = {
    'content': ['unknown', 'ice_action', 'bench_or_penalty_box', 'person_closeup',
                'crowd_or_arena', 'ad_or_fullscreen_graphic', 'studio_or_other', 'mixed_or_transition'],
    'camera_view': ['unknown', 'elevated_wide', 'true_overhead', 'end_or_corner',
                    'low_rinkside', 'tight', 'not_applicable'],
    'temporal_context': ['unknown', 'live', 'replay'],
    'position_usability': ['unknown', 'usable', 'partial', 'unusable'],
    'geometry_status': ['unreviewed', 'insufficient', 'ambiguous', 'accepted', 'rejected'],
}
STATES = ['unreviewed', 'in_progress', 'reviewed']
VISIBILITY = ['visible', 'partially_occluded', 'occluded', 'out_of_frame', 'uncertain']
PROVENANCE = ['manual', 'legacy_import', 'detector_proposal', 'propagated', 'model_inferred']
FEATURE_TYPES = ['landmark', 'blue_line', 'center_line', 'goal_line', 'circle_arc',
                 'crease', 'trapezoid', 'board_intersection', 'unknown']


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_record(entry):
    return dict(schema_version=VERSION, image_id=entry['id'], revision=0,
                shot={key: values[0] for key, values in TAGS.items()},
                reason='', notes='', shot_id=None, leakage_group=None,
                orientation='unknown', review={key: 'unreviewed' for key in ('shot', 'landmarks', 'geometry', 'players')},
                features=[], players=[], ignore_regions=[], proposals=[], calibration=None)


def index_images(source, dataset, source_id, timestamps=None):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    manifest_path = dataset / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        'schema_version': VERSION, 'template': TEMPLATE, 'images': []}
    entries = {entry['id']: entry for entry in manifest['images']}
    timing = read_json(timestamps) if timestamps else {}
    seen, skipped = 0, []
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        if re.search(r'_(v\d+|det|hull|relblue|relred)$', path.stem):
            continue
        relative = path.relative_to(source).as_posix()
        image_id = hashlib.sha256(f'{source_id}/{relative}'.encode()).hexdigest()[:20]
        sha = digest(path)
        previous = entries.get(image_id)
        if previous and previous['sha256'] != sha:
            raise ValueError(f'Source changed: {path}; use a new source ID, preserving old annotations.')
        image = cv2.imread(str(path))
        if image is None:
            skipped.append(str(path))
            continue
        h, w = image.shape[:2]
        stamp = timing.get(relative, {})
        for key in ('timestamp_seconds', 'source_frame_index'):
            value = stamp.get(key)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'Invalid {key} for {relative}')
        entry = dict(id=image_id, path=str(path), relative_path=relative, source_id=source_id,
                     sha256=sha, width=w, height=h, source_video=stamp.get('source_video'),
                     timestamp_seconds=stamp.get('timestamp_seconds'),
                     source_frame_index=stamp.get('source_frame_index'))
        if previous and not stamp:
            for key in ('source_video', 'timestamp_seconds', 'source_frame_index'):
                entry[key] = previous.get(key)
        entries[image_id] = entry
        record_path = dataset / 'records' / (image_id + '.json')
        if not record_path.exists():
            atomic_json(record_path, new_record(entry))
        seen += 1
    manifest['images'] = sorted(entries.values(), key=lambda e: (e['source_id'], e['relative_path']))
    manifest['unreadable_images'] = skipped
    atomic_json(manifest_path, manifest)
    print(f'Indexed {seen} images; dataset total {len(entries)}; unreadable {len(skipped)}')


class Dataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = read_json(self.path / 'manifest.json')
        self.entries = {e['id']: e for e in self.manifest['images']}
        self.lock = threading.RLock()

    def get(self, image_id):
        if image_id not in self.entries:
            raise ValueError('Unknown image ID')
        record = read_json(self.path / 'records' / (image_id + '.json'))
        # Existing geometry review also covers landmark review. Upgrade on edit.
        record['review'].setdefault('landmarks', 'reviewed' if record['review']['geometry'] == 'reviewed' else 'unreviewed')
        for player in record['players']:
            player.setdefault('team', 'official' if player['role'] == 'official' else 'unknown')
            if player['team'] == 'none':
                player['team'] = 'official' if player['role'] == 'official' else 'unknown'
        return record

    def save(self, record):
        with self.lock:
            old = self.get(record['image_id'])
            if old['revision'] != record['revision']:
                raise ValueError('Revision conflict: reload before saving; another editor changed this frame.')
            observed = lambda features: [{k: v for k, v in f.items() if k != 'held_out'} for f in features]
            if observed(record['features']) != observed(old['features']):
                record['review']['landmarks'] = 'in_progress'
            # Any geometry edit invalidates an old calibration and its acceptance.
            if (record['features'] != old['features'] or record['orientation'] != old['orientation']):
                record['calibration'] = None
                if record['shot']['geometry_status'] == 'accepted':
                    record['shot']['geometry_status'] = 'unreviewed'
                record['review']['geometry'] = 'in_progress'
            elif record.get('calibration') != old.get('calibration'):
                raise ValueError('Calibration must be computed by the fit command, not edited directly.')
            validate_record(record, self.entries[record['image_id']])
            record['revision'] += 1
            atomic_json(self.path / 'records' / (record['image_id'] + '.json'), record)
            return record

    def fit(self, image_id, revision=None):
        with self.lock:
            record = self.get(image_id)
            if revision is not None and record['revision'] != revision:
                raise ValueError('Revision conflict: reload before fitting.')
            record['calibration'] = fit_geometry(record)
            record['shot']['geometry_status'] = 'unreviewed'
            record['review']['geometry'] = 'in_progress'
            record['revision'] += 1
            atomic_json(self.path / 'records' / (image_id + '.json'), record)
            return record


def validate_record(record, entry):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def point(p):
        require(isinstance(p, list) and len(p) == 2, 'Point must have two coordinates')
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in p), 'Nonfinite point')
        require(0 <= p[0] < entry['width'] and 0 <= p[1] < entry['height'], 'Point outside image')

    require(record['schema_version'] == VERSION and record['image_id'] == entry['id'], 'Invalid record identity')
    require(record['orientation'] in ('unknown', 'sequence_local', 'rink_fixed'), 'Invalid orientation')
    for field, values in TAGS.items():
        require(record['shot'][field] in values, f'Invalid {field}')
    for field in ('shot', 'geometry', 'players'):
        require(record['review'][field] in STATES, 'Invalid review state')
    require(record['review'].get('landmarks', 'unreviewed') in STATES, 'Invalid landmark review state')
    for key in ('reason', 'notes'):
        require(isinstance(record[key], str), f'{key} must be text')
    for key in ('shot_id', 'leakage_group'):
        require(record[key] is None or isinstance(record[key], str), f'{key} must be text or null')
    ids = set()
    for feature in record['features']:
        require(feature['id'] not in ids, 'Duplicate feature ID')
        ids.add(feature['id'])
        require(feature['type'] in FEATURE_TYPES, 'Invalid feature type')
        require(feature['visibility'] in VISIBILITY, 'Invalid visibility')
        require(feature['provenance'] in PROVENANCE, 'Invalid provenance')
        require(feature.get('landmark') is None or feature['landmark'] in LANDMARKS, 'Unknown landmark')
        require(isinstance(feature.get('held_out', False), bool), 'Invalid held-out flag')
        for p in feature['points']:
            point(p)
        if feature['visibility'] in ('occluded', 'out_of_frame'):
            require(not feature['points'], 'Unseen features must not have invented coordinates')
        elif feature['type'] == 'landmark':
            require(len(feature['points']) == 1, 'Landmark requires one point')
        else:
            require(len(feature['points']) >= 2, 'Polyline requires at least two points')
    for region in record['ignore_regions']:
        require(len(region) >= 3, 'Ignore polygon requires three points')
        for p in region:
            point(p)
    ids = set()
    for player in record['players']:
        require(player['id'] not in ids, 'Duplicate player ID')
        ids.add(player['id'])
        require(player['role'] in ('skater', 'goalie', 'official', 'unknown'), 'Invalid role')
        require(player.get('team', 'unknown') in ('team_a', 'team_b', 'unknown', 'official', 'none'), 'Invalid team')
        require(player['contact_visibility'] in ('both', 'one', 'estimated', 'hidden', 'uncertain'), 'Invalid contact visibility')
        require(player['provenance'] in PROVENANCE, 'Invalid player provenance')
        require(isinstance(player['occluded'], bool) and isinstance(player['truncated'], bool), 'Invalid player flags')
        box = player['bbox']
        require(len(box) == 4, 'Invalid box')
        point(box[:2]); point(box[2:])
        require(box[2] > box[0] and box[3] > box[1], 'Box has no area')
        if player['ice_point'] is not None:
            point(player['ice_point'])
            require(player['contact_visibility'] != 'hidden', 'Hidden feet cannot have an observed contact point')
    if record['shot']['geometry_status'] == 'accepted' and record['review']['geometry'] == 'reviewed':
        calibration = record.get('calibration')
        require(calibration is not None, 'Acceptance requires a calculated fit')
        require(record['orientation'] != 'unknown', 'Acceptance requires documented orientation')
        fingerprint = hashlib.sha256(json.dumps(record['features'], sort_keys=True).encode()).hexdigest()
        require(calibration.get('geometry_fingerprint') == fingerprint, 'Calibration does not match current observations')
    if record['review']['shot'] == 'reviewed':
        require(record['shot']['position_usability'] in ('usable', 'unusable'),
                'Completed usability review requires usable or unusable')


def transform(H, points):
    return cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), np.asarray(H)).reshape(-1, 2)


def fit_geometry(record):
    observations = [f for f in record['features'] if f.get('landmark') in LANDMARKS
                    and f['type'] == 'landmark' and len(f['points']) == 1
                    and f['visibility'] in ('visible', 'partially_occluded')
                    and f['provenance'] in ('manual', 'legacy_import')]
    if len({f['landmark'] for f in observations}) != len(observations):
        raise ValueError('Use each semantic landmark once; duplicates are not independent constraints.')
    fit = [f for f in observations if not f.get('held_out')]
    held = [f for f in observations if f.get('held_out')]
    if len(fit) < 4:
        raise ValueError('Need at least four observed, non-held-out point landmarks.')
    image = np.float64([f['points'][0] for f in fit])
    rink = np.float64([LANDMARKS[f['landmark']] for f in fit])
    for pts in (image, rink):
        if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
            raise ValueError('Degenerate, collinear landmark layout.')
    H, _ = cv2.findHomography(image, rink, 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Could not fit a nonsingular homography.')
    inverse = np.linalg.inv(H)
    validation = []
    for feature in held:
        p, target = feature['points'][0], LANDMARKS[feature['landmark']]
        validation.append(dict(landmark=feature['landmark'],
                               error_px=float(np.linalg.norm(transform(inverse, [target])[0] - p)),
                               error_ft=float(np.linalg.norm(transform(H, [p])[0] - target))))
    hull = cv2.convexHull(np.float32(image)).reshape(-1, 2).tolist()
    return dict(template=TEMPLATE, direction='image_pixels_to_rink_feet', H=H.tolist(),
                fit_landmarks=[f['landmark'] for f in fit], validation=validation,
                validation_count=len(validation), supported_polygon=hull,
                fit_mean_px=float(np.linalg.norm(transform(inverse, rink) - image, axis=1).mean()),
                geometry_fingerprint=hashlib.sha256(json.dumps(record['features'], sort_keys=True).encode()).hexdigest())


def import_legacy(dataset, label_dir, image_dir):
    count, skipped = 0, []
    by_hash = {}
    for entry in dataset.entries.values():
        by_hash.setdefault(entry['sha256'], []).append(entry)
    for path in sorted(Path(label_dir).glob('*.json')):
        image = Path(image_dir) / path.name.removesuffix('.json')
        if not image.exists():
            skipped.append(f'{path.name}: no corresponding source image')
            continue
        matches = by_hash.get(digest(image), [])
        if not matches:
            skipped.append(f'{path.name}: source image hash not in dataset')
            continue
        labels = read_json(path)
        for entry in matches:
            record = dataset.get(entry['id'])
            for name, p in labels.items():
                if name not in LANDMARKS:
                    continue
                proposal = dict(id='legacy-' + hashlib.sha256(f'{path.resolve()}:{name}'.encode()).hexdigest()[:16],
                                type='landmark', landmark=name, points=[p], visibility='visible',
                                provenance='legacy_import', held_out=False, source=str(path.resolve()))
                if not any(f['id'] == proposal['id'] for f in record['proposals'] + record['features']):
                    record['proposals'].append(proposal)
                    count += 1
            dataset.save(record)
    print(f'Imported {count} legacy point proposals; skipped {len(skipped)} files')
    for reason in skipped:
        print(' ', reason)


def propose(dataset, image_id=None, method='center'):
    from vision.detect_center import detect
    from vision.detect import detect as generic_detect
    accepted, abstained = 0, 0
    for entry in dataset.entries.values():
        if image_id and entry['id'] != image_id:
            continue
        record = dataset.get(entry['id'])
        frame = cv2.imread(entry['path'])
        if frame is None:
            raise ValueError('Cannot read source image')
        proposals = []
        if method == 'center':
            try:
                result = detect(frame)
            except ValueError:
                abstained += 1
                continue
            for name, p in result['dots'].items():
                proposals.append(dict(id='center-v3-' + name, type='landmark', landmark=name,
                    points=[p], visibility='visible', provenance='detector_proposal', held_out=False))
        else:
            if record['shot']['content'] != 'ice_action':
                raise ValueError('Tag this frame as ice_action before requesting general line candidates.')
            result = generic_detect(frame)
            for i, line in enumerate(result['lines'][:12]):
                # These are deliberately unassigned paint candidates, not rink identities.
                points = np.asarray([line['p0'], line['p1']]).clip([0, 0], [entry['width']-1, entry['height']-1]).tolist()
                proposals.append(dict(id=f'paint-line-{i}', type='blue_line' if line['color'] == 'blue' else 'unknown',
                    landmark=None, points=points, visibility='visible', provenance='detector_proposal', held_out=False))
        for proposal in proposals:
            fid = proposal['id']
            if not any(f['id'] == fid for f in record['features'] + record['proposals']):
                record['proposals'].append(proposal)
        dataset.save(record)
        accepted += 1
    print(f'{method} proposal frames: {accepted}; abstained: {abstained}. All proposals need review.')


def validate_dataset(dataset):
    errors, counts = [], {}
    for entry in dataset.entries.values():
        try:
            record = dataset.get(entry['id'])
            validate_record(record, entry)
            if not Path(entry['path']).exists() or digest(entry['path']) != entry['sha256']:
                raise ValueError('Source image missing or changed')
            for field, value in record['review'].items():
                counts[f'{field}:{value}'] = counts.get(f'{field}:{value}', 0) + 1
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f'{entry["relative_path"]}: {exc}')
    return dict(images=len(dataset.entries), review_counts=counts, errors=errors,
                missing_timestamps=sum(e['timestamp_seconds'] is None for e in dataset.entries.values()))


def migrate_binary_usability(dataset):
    changed = 0
    for image_id in dataset.entries:
        record = dataset.get(image_id)
        if record['shot']['position_usability'] != 'partial':
            continue
        record.setdefault('migration_history', []).append({
            'schema_change': 'partial_position_usability_to_usable',
            'previous_value': 'partial',
            'reason': 'Binary usability treats any recoverable visible positioning as usable.',
        })
        record['shot']['position_usability'] = 'usable'
        dataset.save(record)
        changed += 1
    print(f'Migrated {changed} partial records to usable; history retained in each record.')


def split_groups(dataset):
    # Union duplicate images, shot groups, and explicitly linked replay groups.
    parent = {key: key for key in dataset.entries}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen, grouped = {}, set()
    for key, entry in dataset.entries.items():
        record = dataset.get(key)
        tokens = [('hash', entry['sha256'])]
        for field in ('shot_id', 'leakage_group'):
            if record[field]:
                tokens.append((field, entry['source_id'], record[field]))
                grouped.add(key)
        for token in tokens:
            if token in seen:
                a, b = root(key), root(seen[token])
                parent[max(a, b)] = min(a, b)
            seen[token] = key
    # If ANY member has unknown grouping, conservatively leave that component unassigned.
    unknown = {root(key) for key in dataset.entries if key not in grouped}
    result = {}
    for key in dataset.entries:
        group = root(key)
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        result[key] = 'unassigned' if group in unknown else ('train' if bucket < 70 else 'validation' if bucket < 85 else 'test')
    return result


def export_dataset(dataset, destination):
    audit = validate_dataset(dataset)
    if audit['errors']:
        raise ValueError('Fix validation errors before export: ' + '; '.join(audit['errors'][:5]))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    splits = split_groups(dataset)
    streams = {key: [] for key in ('shots', 'features', 'players', 'calibrations')}
    collector = {}
    for key, entry in dataset.entries.items():
        record = dataset.get(key)
        base = dict(image=entry, split=splits[key], shot_id=record['shot_id'], leakage_group=record['leakage_group'],
                    orientation=record['orientation'], notes=record['notes'], geometry_status=record['shot']['geometry_status'])
        if record['review']['shot'] == 'reviewed':
            streams['shots'].append(dict(base, labels=record['shot'], reason=record['reason']))
        if record['review'].get('landmarks') == 'reviewed' or record['review']['geometry'] == 'reviewed':
            observed = [f for f in record['features'] if f['provenance'] in ('manual', 'legacy_import')]
            streams['features'].append(dict(base, features=observed, ignore_regions=record['ignore_regions'], orientation=record['orientation']))
            collector[key] = {f['landmark']: f['points'][0] for f in observed
                              if f['type'] == 'landmark' and f.get('landmark') and len(f['points']) == 1
                              and f['visibility'] in ('visible', 'partially_occluded')}
        calibration = record.get('calibration')
        accepted = (record['shot']['geometry_status'] == 'accepted'
                    and record['review']['geometry'] == 'reviewed' and calibration)
        if accepted:
            streams['calibrations'].append(dict(base, calibration=calibration, orientation=record['orientation']))
        if record['review']['players'] == 'reviewed':
            players = copy.deepcopy(record['players'])
            for player in players:
                player['rink_point_derived'] = None
                p = player['ice_point']
                if accepted and p and player['contact_visibility'] in ('both', 'one'):
                    hull = np.float32(calibration['supported_polygon'])
                    if cv2.pointPolygonTest(hull, tuple(p), False) >= 0:
                        player['rink_point_derived'] = transform(calibration['H'], [p])[0].tolist()
            streams['players'].append(dict(base, players=players, ignore_regions=record['ignore_regions'],
                calibration_fingerprint=calibration['geometry_fingerprint'] if accepted else None))
    for key, rows in streams.items():
        target = destination / (key + '.jsonl')
        temporary = target.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        os.replace(temporary, target)
    atomic_json(destination / 'collector.json', collector)
    atomic_json(destination / 'audit.json', dict(audit, export_counts={k: len(v) for k, v in streams.items()},
                                                template=TEMPLATE, splits=splits))
    print(json.dumps({k: len(v) for k, v in streams.items()}))


def contact_sheets(dataset, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = list(dataset.entries.values())
    for offset in range(0, len(entries), 48):
        sheet = np.zeros((6 * 160, 8 * 256, 3), np.uint8)
        for index, entry in enumerate(entries[offset:offset+48]):
            image = cv2.imread(entry['path'])
            h, w = image.shape[:2]
            factor = min(256 / w, 136 / h)
            thumb = cv2.resize(image, (round(w * factor), round(h * factor)))
            y, x = divmod(index, 8)
            sheet[y*160:y*160+thumb.shape[0], x*256:x*256+thumb.shape[1]] = thumb
            cv2.putText(sheet, entry['relative_path'][-32:], (x*256+4, y*160+153),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.imwrite(str(output / f'sheet_{offset//48+1:03}.jpg'), sheet)


def sample_timed_frames(dataset, output, rates):
    """Select existing, timestamped frames, without inventing missing samples."""
    groups, skipped = {}, []
    for entry in dataset.entries.values():
        record = dataset.get(entry['id'])
        if entry['timestamp_seconds'] is None or not record['shot_id']:
            skipped.append(entry['id'])
            continue
        groups.setdefault((entry['source_id'], record['shot_id']), []).append(entry)
    selections = {}
    for rate in rates:
        rows = []
        for (source, shot), entries in sorted(groups.items()):
            entries.sort(key=lambda e: e['timestamp_seconds'])
            times = np.array([e['timestamp_seconds'] for e in entries])
            # A fixed origin lets the 2 Hz targets coincide with every other 4 Hz target.
            targets = np.arange(math.ceil(times[0] * rate), math.floor(times[-1] * rate) + 1) / rate
            used = set()
            for target in targets:
                i = int(np.abs(times - target).argmin())
                if abs(times[i] - target) > .25 / rate or i in used:
                    continue
                used.add(i)
                rows.append(dict(image_id=entries[i]['id'], source_id=source, shot_id=shot,
                                 target_seconds=float(target), timestamp_seconds=float(times[i])))
        selections[str(rate)] = rows
    atomic_json(output, dict(rates_hz=rates, selections=selections, skipped_missing_timing_or_shot=skipped,
                             note='Existing frames only; targets without a frame within a quarter interval are omitted.'))
    print(json.dumps({rate: len(rows) for rate, rows in selections.items()}))


def serve(dataset, port, mode):
    web = Path(__file__).parent / 'web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, content_type='application/json'):
            data = json.dumps(payload).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/api/manifest':
                    with dataset.lock:
                        dataset.manifest = read_json(dataset.path / 'manifest.json')
                        dataset.entries = {e['id']: e for e in dataset.manifest['images']}
                    summaries = {key: dataset.get(key)['review'] for key in dataset.entries}
                    self.send(200, dict(dataset.manifest, tags=TAGS, landmarks=LANDMARKS,
                        feature_types=FEATURE_TYPES, visibility=VISIBILITY, summaries=summaries, mode=mode))
                elif path.startswith('/api/record/'):
                    self.send(200, dataset.get(path.rsplit('/', 1)[1]))
                elif path.startswith('/image/'):
                    entry = dataset.entries[path.rsplit('/', 1)[1]]
                    kind = 'image/png' if Path(entry['path']).suffix.lower() == '.png' else 'image/jpeg'
                    self.send(200, Path(entry['path']).read_bytes(), kind)
                elif path in ('/', '/app.js', '/style.css'):
                    name = 'index.html' if path == '/' else path[1:]
                    kind = {'index.html': 'text/html', 'app.js': 'text/javascript', 'style.css': 'text/css'}[name]
                    self.send(200, (web / name).read_bytes(), kind)
                else:
                    self.send(404, {'error': 'Not found'})
            except (ValueError, KeyError, OSError) as exc:
                self.send(400, {'error': str(exc)})

        def do_POST(self):
            # Only the local page may mutate data, not cross-origin browser requests.
            origin = self.headers.get('Origin')
            expected = f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected.removeprefix('http://') or origin not in (None, expected):
                self.send(403, {'error': 'Use the local 127.0.0.1 URL'})
                return
            if self.headers.get('Content-Type') != 'application/json':
                self.send(415, {'error': 'JSON required'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if self.path == '/api/save':
                    self.send(200, dataset.save(data))
                elif self.path == '/api/fit':
                    self.send(200, dataset.fit(data['image_id'], data['revision']))
                elif self.path == '/api/propose':
                    with dataset.lock:
                        record = dataset.get(data['image_id'])
                        if record['revision'] != data['revision']:
                            raise ValueError('Revision conflict: reload before proposing.')
                        if data['method'] not in ('center', 'general'):
                            raise ValueError('Unknown proposal method')
                        propose(dataset, data['image_id'], data['method'])
                        self.send(200, dataset.get(data['image_id']))
                else:
                    self.send(404, {'error': 'Not found'})
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {'error': str(exc)})
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Open http://127.0.0.1:{server.server_port} — Ctrl+C to stop', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index')
    index.add_argument('source')
    index.add_argument('--dataset', required=True)
    index.add_argument('--source-id', required=True)
    index.add_argument('--timestamps', help='JSON mapping relative filenames to explicit source timing')
    for name in ('review-shots', 'review-geometry', 'review-players', 'validate', 'export', 'contact-sheet', 'import-legacy', 'propose', 'sample', 'migrate-binary-usability'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True)
        if name.startswith('review-'):
            p.add_argument('--port', type=int, default=8765)
        if name in ('export', 'contact-sheet', 'sample'):
            p.add_argument('--out', required=True)
        if name == 'sample':
            p.add_argument('--hz', nargs='+', type=int, choices=[2, 4], default=[2, 4])
        if name == 'import-legacy':
            p.add_argument('--labels', required=True)
            p.add_argument('--images', required=True)
        if name == 'propose':
            p.add_argument('--image-id', required=True, help='One manifest image ID; proposals are deliberately opt-in per frame')
            p.add_argument('--method', choices=['center', 'general'], default='center')
    args = parser.parse_args()
    try:
        if args.command == 'index':
            index_images(args.source, args.dataset, args.source_id, args.timestamps)
            return
        dataset = Dataset(args.dataset)
        if args.command.startswith('review-'):
            serve(dataset, args.port, args.command.removeprefix('review-'))
        elif args.command == 'validate':
            audit = validate_dataset(dataset)
            print(json.dumps(audit, indent=2))
            if audit['errors']:
                raise SystemExit(1)
        elif args.command == 'export':
            export_dataset(dataset, args.out)
        elif args.command == 'contact-sheet':
            contact_sheets(dataset, args.out)
        elif args.command == 'sample':
            sample_timed_frames(dataset, args.out, args.hz)
        elif args.command == 'migrate-binary-usability':
            migrate_binary_usability(dataset)
        elif args.command == 'import-legacy':
            import_legacy(dataset, args.labels, args.images)
        elif args.command == 'propose':
            if args.image_id not in dataset.entries:
                raise ValueError('Unknown image ID')
            propose(dataset, args.image_id, args.method)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
