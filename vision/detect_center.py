"""Detect conservative center-ice geometry for labeling proposals and diagnostics.

python -m vision.detect_center frames/601.png

No clicked labels are read. E/W mean image-right/image-left, as in this shot's
labels; this is not an automatic determination of the teams' attacking ends.
Outputs an overlay, observed dot coordinates, and a diagnostic geometry report.
The four-dot homography is local: do not extrapolate it to the boards unchecked.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from vision.detect import detect as baseline, _seg_dist


NAMES = ['dotNZ_W_N', 'dotNZ_E_N', 'dotNZ_W_S', 'dotNZ_E_S']
RINK_DOTS = np.float32([[-20, 22], [20, 22], [-20, -22], [20, -22]])


def project(H, points):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)


def dot_candidates(frame, det):
    scale = frame.shape[1] / 1920
    # Broadcast dots are pale, wide ellipses, not image-space circles.
    mask = ((det['relred'] > 8) & (frame.min(axis=2) > 100)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white = ((frame.max(axis=2) - frame.min(axis=2) < 40)
             & (frame.min(axis=2) > 145))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if not (120 * scale**2 < area < 2500 * scale**2
                and 1.6 < w / h < 6 and area / (w * h) > .6
                and len(contour) >= 5):
            continue
        (cx, cy), (ew, eh), angle = cv2.fitEllipse(contour)
        ring = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(ring, ((cx, cy), (ew * 1.7, eh * 2.2), angle), 1, -1)
        cv2.ellipse(ring, ((cx, cy), (ew * 1.15, eh * 1.3), angle), 0, -1)
        if not ring.any() or white[ring > 0].mean() < .9:
            continue
        moments = cv2.moments(contour)
        candidates.append([moments['m10'] / area, moments['m01'] / area])
    return np.asarray(candidates, np.float32).reshape(-1, 2)


def detect(frame):
    det = baseline(frame)
    blues = [line for line in det['lines'] if line['color'] == 'blue']
    if len(blues) != 2:
        raise ValueError('No reliable center-ice fix: expected two blue lines.')
    blues.sort(key=lambda line: line['point'][0])
    candidates = dot_candidates(frame, det)
    scale = frame.shape[1] / 1920
    solutions = []
    for four in itertools.combinations(candidates, 4):
        points = np.asarray(sorted(four, key=lambda p: p[1]), np.float32)
        points[:2] = points[:2][np.argsort(points[:2, 0])]
        points[2:] = points[2:][np.argsort(points[2:, 0])]
        if (points[2:, 1].min() - points[:2, 1].max() < 100 * scale
                or min(points[1, 0] - points[0, 0], points[3, 0] - points[2, 0]) < 250 * scale):
            continue
        H = cv2.getPerspectiveTransform(RINK_DOTS, points)
        # Four points always fit exactly. Validate on independent BLUE paint,
        # never on the four fitted dots themselves.
        errors = []
        for x, blue in zip((-25, 25), blues):
            samples = project(H, [(x, y) for y in np.linspace(-22, 22, 25)])
            errors.extend(_seg_dist(p, blue) for p in samples)
        error = float(np.mean(errors))
        if max(errors) < 45 * scale:
            solutions.append((error, H, points))
    if not solutions:
        raise ValueError('No reliable center-ice fix: dot geometry does not match blue lines.')
    solutions.sort(key=lambda s: s[0])
    if len(solutions) > 1 and solutions[1][0] - solutions[0][0] < 5 * scale:
        raise ValueError('Ambiguous center-ice fix: competing dot configurations.')
    error, H, points = solutions[0]

    # Independently locate the red center stripe near the dot-model prediction.
    # Row medians tolerate stripe gaps, logos, and partially occluding players.
    h, w = frame.shape[:2]
    expected = project(H, [(0, 40), (0, -40)])
    rows = []
    for y in range(h):
        x = np.interp(y, expected[:, 1], expected[:, 0])
        lo, hi = max(0, int(x - 28 * scale)), min(w, int(x + 28 * scale + 1))
        xs = np.flatnonzero((det['relred'][y, lo:hi] > 14)
                            & (frame[y, lo:hi].min(axis=1) > 110)) + lo
        if 5 * scale <= len(xs) <= 40 * scale:
            rows.append([float(np.median(xs)), y])
    if len(rows) < h * .25:
        raise ValueError('No reliable center-ice fix: insufficient red stripe support.')
    rows = np.float32(rows)
    vx, vy, x0, y0 = cv2.fitLine(rows, cv2.DIST_WELSCH, 0, .01, .01).ravel()
    if abs(vy) < .5:
        raise ValueError('Invalid center stripe orientation.')
    ys = np.percentile(rows[:, 1], [1, 99])
    red = np.column_stack((x0 + (ys - y0) * vx / vy, ys))
    center = project(H, [(0, 0)])[0]
    theta = np.linspace(0, 2 * np.pi, 241)
    circle = project(H, np.column_stack((15 * np.cos(theta), 15 * np.sin(theta))))
    return dict(dots=dict(zip(NAMES, points.tolist())), blue_lines=blues,
                red_line=red.tolist(), center=center.tolist(), circle=circle,
                H=H.tolist(), blue_validation_mean_px=error,
                candidate_count=len(candidates))


def overlay(frame, result):
    out = frame.copy()
    cyan, yellow, magenta = (255, 210, 40), (30, 240, 255), (220, 80, 255)
    def label(text, position, color):
        p = tuple(np.int32(position))
        cv2.putText(out, text, p, cv2.FONT_HERSHEY_SIMPLEX, .6, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(out, text, p, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
    for name, line in zip(('W blue line', 'E blue line'), result['blue_lines']):
        cv2.line(out, tuple(np.int32(line['p0'])), tuple(np.int32(line['p1'])), cyan, 2, cv2.LINE_AA)
        mid = (np.asarray(line['p0']) + line['p1']) / 2
        label(name, mid + [15, 0], cyan)
    a, b = np.int32(result['red_line'])
    cv2.line(out, tuple(a), tuple(b), magenta, 2, cv2.LINE_AA)
    label('Center red line', a + [25, 130], magenta)
    for name, point in result['dots'].items():
        p = np.int32(point)
        cv2.circle(out, tuple(p), 12, yellow, 2, cv2.LINE_AA)
        cv2.drawMarker(out, tuple(p), yellow, cv2.MARKER_CROSS, 8, 1)
        offset = [20, -23] if p[0] < out.shape[1] * .5 else [-160, -23]
        label(name, p + offset, yellow)
    circle = np.int32(result['circle'])
    for i in range(0, len(circle) - 3, 6):
        cv2.polylines(out, [circle[i:i+4]], False, yellow, 2, cv2.LINE_AA)
    center = np.int32(result['center'])
    cv2.drawMarker(out, tuple(center), yellow, cv2.MARKER_DIAMOND, 18, 2)
    label('Center (inferred)', center + [24, -18], yellow)
    cv2.rectangle(out, (12, 12), (1190, 68), (25, 25, 25), -1)
    label('v3 | 2 blue lines + center stripe | 4 observed NZ dots', (26, 35), (255, 255, 255))
    label('Dashed circle / diamond: inferred from dot geometry; not independent detections', (26, 58), yellow)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f'Cannot read {args.image}')
    try:
        result = detect(frame)
    except ValueError as exc:
        parser.exit(2, str(exc) + '\n')
    stem = args.image.with_name(args.image.stem + '_v3')
    cv2.imwrite(str(stem.with_suffix('.png')), overlay(frame, result))
    stem.with_suffix('.labels.json').write_text(json.dumps(result['dots'], indent=2) + '\n')
    report = {k: v for k, v in result.items() if k not in ('blue_lines', 'circle')}
    report['blue_lines'] = [{'p0': l['p0'], 'p1': l['p1']} for l in result['blue_lines']]
    report['inferred_features'] = ['center', 'center_circle_radius_15ft']
    report['homography_direction'] = 'rink_feet_to_image_pixels'
    report['scope'] = 'Center-ice shot with four visible NZ dots; geometry tuned on 601. No board intersections inferred.'
    stem.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{stem}.png: 4 dots, 3 lines; blue-line validation mean {result["blue_validation_mean_px"]:.1f}px')


if __name__ == '__main__':
    main()
