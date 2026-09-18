/* Browser client for reviewing shots, rink geometry, calibration, and players. */
'use strict';
const $ = id => document.getElementById(id);
let manifest, record, entry, index = 0, image = new Image();
let dirty = false, saving = false, savePromise = null, loading = false, editVersion = 0, timer, undoStack = [], selection = null;
let zoom = 1, offset = [0, 0], draft = [], drag = null;
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const clone = value => JSON.parse(JSON.stringify(value));
const uid = () => crypto.randomUUID();
const message = text => { $('errors').textContent = text; };
const PHASES = {
  shots: { key: 'shot', title: 'Choose usable or unusable', instruction: 'First pass: make one decision. A frame is usable when it shows enough rink context to recover positions for at least some visible on-ice people. Either choice saves and advances.', done: 'Done →', tools: ['inspect'] },
  geometry: { key: 'landmarks', title: 'Mark visible rink landmarks', instruction: 'For selected ice views, choose a named point on the diagram and click it in the image. Start with faceoff dots. Add clear line/board intersections if visible. You do not need to fill every landmark.', done: 'Landmarks checked →', tools: ['point', 'move', 'inspect'] },
  calibration: { key: 'geometry', title: 'Check the rink fit', instruction: 'Use the points from phase 2 to check alignment. The overlay is calculated automatically. Inspect its alignment; extra check points are optional. A view can finish as insufficient or ambiguous.', done: 'Save fit review →', tools: ['inspect'] },
  players: { key: 'players', title: 'Mark anonymous player positions', instruction: 'For selected ice views, drag a box around every visible on-ice person. Add the skate-contact point when visible and label officials separately. Finish after checking the whole frame.', done: 'Players checked →', tools: ['box', 'contact', 'inspect'] }
};
const TOOL_NAMES = { inspect: 'Pan / inspect', point: 'Place a landmark', move: 'Move selected landmark', box: 'Draw a person box', contact: 'Place selected ice point', line: 'Trace a line / arc', ignore: 'Outline an area to ignore' };
function landmarkName(name) {
  if (!name) return 'Choose a point…'; if (name === 'center') return 'Center-ice dot';
  if (name === 'centerBrd_N') return 'Center line at north-side boards';
  if (name === 'centerBrd_S') return 'Center line at south-side boards';
  const [kind, end, side] = name.split('_');
  const names = { dotNZ: 'Neutral-zone dot', dotEZ: 'End-zone dot', blue: 'Blue line at boards', goalBrd: 'Goal line at boards', post: 'Goalpost base on ice', trapGL: 'Trapezoid corner at goal line', trapEB: 'Trapezoid corner at end boards' };
  return `${names[kind] || kind} · ${end} / ${side}`;
}
function setTool(tool) {
  const allowed = [...PHASES[$('mode').value].tools]; if (!allowed.includes(tool)) allowed.push(tool);
  options($('tool'), allowed); for (const o of $('tool').options) o.textContent = TOOL_NAMES[o.value]; $('tool').value = tool;
  draft = []; drag = null; render();
}
function phaseUI() {
  const mode = $('mode').value, phase = PHASES[mode], tool = $('tool').value;
  document.querySelectorAll('[data-phase]').forEach(b => { b.setAttribute('aria-current', b.dataset.phase === mode ? 'step' : 'false'); });
  $('phaseTitle').textContent = phase.title; $('phaseInstructions').textContent = phase.instruction;
  for (const [id, show] of Object.entries({ shotPanel: mode === 'shots', geometryPanel: mode === 'geometry', calibrationPanel: mode === 'calibration', playerPanel: mode === 'players', annotationPanel: mode !== 'shots', suggestionPanel: mode === 'geometry', proposalToggle: mode === 'geometry', toolControl: mode === 'geometry' || mode === 'players', newAnnotation: mode === 'geometry' || mode === 'players', finish: tool === 'line' || tool === 'ignore', sequenceDetails: false, completionPanel: mode !== 'shots' })) $(id).hidden = !show;
  $('newAnnotation').textContent = mode === 'players' ? 'Add another person' : 'Add another landmark';
  $('annotationTitle').textContent = mode === 'players' ? 'People in this frame' : mode === 'calibration' ? 'Select a point to reserve for checking' : 'Marked landmarks';
  $('heldOut').disabled = selection?.kind !== 'features' || !selected()?.landmark;
  $('phaseStatus').textContent = record.review[phase.key] === 'reviewed' ? 'This phase is complete for this frame.' : 'This frame is unfinished in this phase.';
  $('completeNext').textContent = phase.done;
  $('completionHelp').textContent = mode === 'geometry' ? 'Finish with the visible points you can identify. You can check the rink fit in phase 3 later.' : mode === 'players' ? 'A checked frame with no on-ice people is a valid empty annotation.' : 'Select a result above before finishing. Acceptance needs a calculated fit and visual inspection. Check points are optional.';
  $('canvasHint').textContent = mode === 'shots' ? 'Wheel to zoom · drag to pan. No drawing is needed in this phase.' : mode === 'calibration' ? 'Inspect the green overlay against the paint · A accepts · D rejects · I marks insufficient · Enter saves and continues.' : mode === 'geometry' ? 'Choose a point, then click the image · M moves the selected point · Enter starts the next landmark · wheel to zoom · Escape cancels a trace.' : '1–9 select a person · 0 selects person 10 · Backspace deletes selected person · A = team_a · B = team_b · S = skater · G = goalie · R = official · applies to the selected person or the next box · wheel to zoom.';
  $('markUsable').classList.toggle('selected', record.shot.position_usability === 'usable');
  $('markUnusable').classList.toggle('selected', record.shot.position_usability === 'unusable');
  $('landmarkHelp').textContent = $('landmark').value ? `${landmarkName($('landmark').value)}. E/W and N/S refer to the diagram’s axes, not screen directions.` : 'Pick a dot or a clearly visible intersection. The diagram also lets you choose by clicking a point.';
  $('suggestionSummary').textContent = `Suggested landmarks to check (${record.proposals.length})`;
  for (const key of ['shot', 'landmarks', 'geometry', 'players']) $('review_' + key).parentElement.hidden = key !== phase.key;
}
async function api(path, body) {
  const result = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await result.json();
  if (!result.ok) throw Error(data.error || result.statusText);
  return data;
}
function options(element, values) {
  element.replaceChildren(...values.map(value => {
    const option = document.createElement('option'); option.value = value; option.textContent = value.replaceAll('_', ' '); return option;
  }));
}
function changed() {
  dirty = true; editVersion++; $('saveState').textContent = 'Unsaved changes';
  clearTimeout(timer); timer = setTimeout(() => save().catch(e => message(e.message)), 1000);
}
function mutate(fn) {
  if (loading) return;
  undoStack.push(clone(record)); if (undoStack.length > 40) undoStack.shift();
  fn();
  const before = undoStack[undoStack.length - 1];
  if (JSON.stringify(before.features) !== JSON.stringify(record.features) || before.orientation !== record.orientation) { record.calibration = null; record.shot.geometry_status = 'unreviewed'; record.review.geometry = 'in_progress'; }
  const observed = fs => fs.map(({ held_out, ...f }) => f);
  if (JSON.stringify(observed(before.features)) !== JSON.stringify(observed(record.features))) record.review.landmarks = 'in_progress';
  if (JSON.stringify(before.players) !== JSON.stringify(record.players)) record.review.players = 'in_progress';
  changed(); render();
}
async function save() {
  clearTimeout(timer);
  if (savePromise) {
    await savePromise;
    if (dirty) return save();
    return;
  }
  if (!dirty) return;
  saving = true; $('saveState').textContent = 'Saving…';
  const version = editVersion;
  const snapshot = clone(record);
  savePromise = (async () => {
    let succeeded = false;
    try {
      const result = await api('/api/save', snapshot);
      succeeded = true;
      if (version === editVersion) { record = result; dirty = false; render(); }
      else { record.revision = result.revision; }
      manifest.summaries[record.image_id] = clone(record.review);
      $('saveState').textContent = dirty ? 'Unsaved changes' : 'Saved';
      message('');
      if (!dirty && !loading && $('mode').value === 'calibration' && !record.calibration) {
        loading = true;
        try { await autoFit(); } finally { loading = false; }
      }
    } catch (e) { $('saveState').textContent = `Save failed: ${e.message}`; throw e; }
    finally {
      saving = false; savePromise = null;
      if (succeeded && dirty) timer = setTimeout(() => save().catch(e => message(e.message)), 1000);
    }
  })();
  return savePromise;
}
function modeKey() { return PHASES[$('mode').value].key; }
function filtered() {
  const query = $('search').value.toLowerCase();
  const pilot = new Set(manifest.selections?.geometry_pilot_v1 || []);
  return manifest.images.map((e, i) => ({ e, i })).filter(({ e }) => e.relative_path.toLowerCase().includes(query)
    && (!$('geometryPilot').checked || pilot.has(e.id))
    && (!$('unreviewed').checked || manifest.summaries[e.id][modeKey()] !== 'reviewed'));
}
function frameMenu() {
  $('frame').replaceChildren(...filtered().map(({ e, i }) => {
    const option = document.createElement('option'); option.value = i; option.textContent = `${e.source_id} / ${e.relative_path}`; return option;
  }));
  $('frame').value = String(index);
  const pilot = new Set(manifest.selections?.geometry_pilot_v1 || []);
  const scope = $('geometryPilot').checked ? manifest.images.filter(e => pilot.has(e.id)) : manifest.images;
  const completed = scope.filter(e => manifest.summaries[e.id][modeKey()] === 'reviewed').length;
  $('progress').textContent = `${completed} / ${scope.length} complete in this phase`;
}
async function load(i) {
  if (loading) return;
  await save();
  loading = true;
  try {
    const nextEntry = manifest.images[i];
    const nextRecord = await api('/api/record/' + nextEntry.id);
    const nextImage = new Image(); nextImage.src = '/image/' + nextEntry.id; await nextImage.decode();
    index = i; entry = nextEntry; record = nextRecord; image = nextImage; localStorage.setItem('ice-last-image', entry.id);
    $('completionFeedback').textContent = '';
    selection = null; draft = []; undoStack = []; fitView(); render(); frameMenu(); $('saveState').textContent = 'Saved';
    await autoFit();
  } finally { loading = false; }
}
// Called while loading is locked so edits cannot race with the fit response.
async function autoFit() {
  if ($('mode').value !== 'calibration' || !record || record.review.geometry === 'reviewed') return;
  if (record.calibration && record.orientation !== 'unknown') return;
  $('calibration').textContent = 'Calculating rink overlay…';
  try {
    const result = record.shot.geometry_status;
    if (record.orientation === 'unknown') {
      const snapshot = clone(record);
      snapshot.orientation = 'sequence_local';
      snapshot.calibration = null;
      snapshot.shot.geometry_status = 'unreviewed';
      snapshot.review.geometry = 'in_progress';
      record = await api('/api/save', snapshot);
      render();
    }
    record = await api('/api/fit', { image_id: record.image_id, revision: record.revision });
    if (result !== 'unreviewed') { record.shot.geometry_status = result; changed(); }
    manifest.summaries[record.image_id] = clone(record.review);
    render(); frameMenu();
  } catch (e) {
    $('calibration').textContent = `Overlay unavailable: ${e.message}`;
  }
}
async function navigate(delta) {
  await save();
  const items = filtered().map(x => x.i);
  const target = delta > 0 ? items.find(i => i > index) : items.slice().reverse().find(i => i < index);
  if (target !== undefined) await load(target);
}
function selected() {
  return selection && record[selection.kind][selection.index];
}
function render() {
  if (!record) return;
  $('imageInfo').textContent = `${entry.source_id} / ${entry.relative_path} · ${entry.width}×${entry.height} · ${entry.timestamp_seconds === null ? 'timing unknown' : entry.timestamp_seconds + 's'} · revision ${record.revision}`;
  $('shotId').value = record.shot_id || ''; $('leakageGroup').value = record.leakage_group || '';
  $('notes').value = record.notes;
  for (const key of Object.keys(manifest.tags)) {
    const control = $('tag_' + key); if (control) control.value = record.shot[key];
  }
  for (const key of ['shot', 'landmarks', 'geometry', 'players']) $('review_' + key).value = record.review[key];
  $('orientation').value = record.orientation; $('geometryStatus').value = record.shot.geometry_status;
  const mode = $('mode').value;
  const annotation = selected();
  if (selection?.kind === 'features' && annotation) {
    $('landmark').value = annotation.landmark || ''; $('featureType').value = annotation.type;
    $('visibility').value = annotation.visibility; $('heldOut').checked = !!annotation.held_out;
  }
  if (selection?.kind === 'players' && annotation) {
    $('role').value = annotation.role; $('team').value = annotation.team || (annotation.role === 'official' ? 'official' : 'unknown'); $('contactVisibility').value = annotation.contact_visibility;
    $('occluded').checked = annotation.occluded; $('truncated').checked = annotation.truncated;
  }
  $('annotations').replaceChildren();
  const kinds = mode === 'shots' ? [] : mode === 'players' ? ['players', 'ignore_regions'] : mode === 'calibration' ? ['features'] : ['features', 'ignore_regions'];
  for (const kind of kinds) record[kind].forEach((item, i) => {
    const div = document.createElement('div'); div.className = 'item' + (selection?.kind === kind && selection.index === i ? ' selected' : '');
    div.textContent = kind === 'features' ? `${item.landmark ? landmarkName(item.landmark) : item.type.replaceAll('_', ' ')} · ${item.visibility}${item.held_out ? ' · check point' : ''}` :
      kind === 'players' ? `${item.role} ${i + 1} · ${item.team || 'unknown'} · ${item.ice_point ? 'ice point' : 'no ice point'}` : `Ignore polygon ${i + 1}`;
    div.onclick = () => { selection = { kind, index: i }; render(); }; $('annotations').append(div);
  });
  $('proposals').replaceChildren();
  record.proposals.forEach((proposal, i) => {
    const div = document.createElement('div'); div.className = 'item';
    const text = document.createElement('span'); text.textContent = `${proposal.landmark ? landmarkName(proposal.landmark) : proposal.type.replaceAll('_', ' ')} `; div.append(text);
    for (const accept of [true, false]) {
      const button = document.createElement('button'); button.textContent = accept ? 'Correct — keep' : 'Incorrect — discard';
      button.onclick = () => mutate(() => {
        const p = record.proposals.splice(i, 1)[0];
        if (accept) { p.original_provenance = p.provenance; p.provenance = 'manual'; record.features.push(p); record.review.geometry = 'in_progress'; selection = { kind: 'features', index: record.features.length - 1 }; }
      }); div.append(button);
    } $('proposals').append(div);
  });
  const c = record.calibration;
  $('calibration').textContent = c ? `Fit: ${c.fit_landmarks.length} points, ${c.fit_mean_px.toFixed(2)} px mean\nHeld out: ${c.validation_count}\n` +
    c.validation.map(v => `${v.landmark}: ${v.error_px.toFixed(1)} px / ${v.error_ft.toFixed(2)} ft`).join('\n') : 'No calibration. Need ≥4 fit landmarks; held-out check points are optional.';
  phaseUI(); drawRink(); drawPlayerRink(); draw();
}
function drawPlayerRink() {
  const c = $('playerRink').getContext('2d'); c.clearRect(0, 0, 330, 165);
  const map = ([x, y]) => [165 + x * 1.5, 82.5 - y * 1.5];
  const line = (points, color, width = 1) => { c.strokeStyle = color; c.lineWidth = width; c.beginPath(); points.forEach((p, i) => i ? c.lineTo(...map(p)) : c.moveTo(...map(p))); c.stroke(); };
  const circle = (x, y, r, color) => { c.strokeStyle = color; c.lineWidth = 1; c.beginPath(); c.arc(...map([x, y]), r * 1.5, 0, Math.PI * 2); c.stroke(); };
  c.fillStyle = '#e5edf0'; c.strokeStyle = '#71818a'; c.lineWidth = 1;
  c.beginPath(); c.roundRect(15, 18.75, 300, 127.5, 42); c.fill(); c.stroke();
  for (const x of [-89, -25, 0, 25, 89]) line([[x, 42.5], [x, -42.5]], Math.abs(x) === 25 ? '#2663c5' : '#c44545', Math.abs(x) === 25 ? 2 : 1);
  for (const [x, y] of [[0, 0], [-69, 22], [-69, -22], [69, 22], [69, -22]]) circle(x, y, 15, x === 0 ? '#2663c5' : '#c44545');
  for (const [name, point] of Object.entries(manifest.landmarks)) if (name === 'center' || name.startsWith('dot')) { c.fillStyle = name === 'center' ? '#2663c5' : '#c44545'; c.beginPath(); c.arc(...map(point), 2, 0, Math.PI * 2); c.fill(); }
  for (const end of [-1, 1]) {
    for (const side of [-1, 1]) line([[end * 89, side * 11], [end * 100, side * 14]], '#c44545');
    const crease = Array.from({ length: 33 }, (_, i) => { const t = -Math.PI / 2 + i * Math.PI / 32; return [end * (89 - 6 * Math.cos(t)), 6 * Math.sin(t)]; });
    line(crease, '#c44545');
  }
  const calibration = record.calibration;
  if (!calibration || record.shot.geometry_status !== 'accepted') { $('playerRinkStatus').textContent = 'Accept a checked calibration to preview derived rink positions.'; return; }
  let count = 0, estimated = 0, boxEstimates = 0, offMap = 0, invalid = 0;
  record.players.forEach((player, i) => {
    const fallback = !player.ice_point;
    const p = player.ice_point || [(player.bbox[0] + player.bbox[2]) / 2, player.bbox[3]];
    const xy = projection(calibration.H, p);
    if (!xy.every(Number.isFinite)) { invalid++; return; }
    const raw = map(xy), pos = [Math.max(6, Math.min(324, raw[0])), Math.max(6, Math.min(159, raw[1]))];
    const clipped = pos.some((v, j) => v !== raw[j]); if (clipped) offMap++;
    const approximate = fallback || !['both', 'one'].includes(player.contact_visibility);
    if (fallback) boxEstimates++; else if (approximate) estimated++;
    c.strokeStyle = c.fillStyle = player.role === 'official' ? '#634ba0' : player.team === 'team_a' ? '#007d92' : player.team === 'team_b' ? '#c16b00' : '#555';
    c.lineWidth = 2; c.beginPath();
    if (fallback || clipped) { c.moveTo(pos[0], pos[1] - 5); c.lineTo(pos[0] + 5, pos[1]); c.lineTo(pos[0], pos[1] + 5); c.lineTo(pos[0] - 5, pos[1]); c.closePath(); }
    else c.arc(...pos, 4, 0, Math.PI * 2);
    if (!approximate) c.fill(); c.stroke();
    if (selection?.kind === 'players' && selection.index === i) { c.strokeStyle = '#111'; c.strokeRect(pos[0] - 7, pos[1] - 7, 14, 14); }
    c.font = '11px system-ui'; c.fillText(String(i + 1), pos[0] > 305 ? pos[0] - 17 : pos[0] + 6, pos[1] < 14 ? pos[1] + 13 : pos[1] - 3); count++;
  });
  $('playerRinkStatus').textContent = `${count}/${record.players.length} positions · ${estimated} estimated contacts · ${boxEstimates} box-bottom estimates · ${offMap} off-map (shown at edge)${invalid ? ` · ${invalid} cannot be projected` : ''}. Teal: team_a; orange: team_b; purple: officials. Hollow: estimated; diamond: box estimate or off-map. Box estimates are preview-only; no contact labels are changed.`;
}
function resize() { const box = $('viewport').getBoundingClientRect(); canvas.width = Math.floor(box.width); canvas.height = Math.floor(box.height); draw(); }
function fitView() { resize(); zoom = Math.min(canvas.width / image.width, canvas.height / image.height); offset = [(canvas.width - image.width * zoom) / 2, (canvas.height - image.height * zoom) / 2]; draw(); }
function screen(p) { return [p[0] * zoom + offset[0], p[1] * zoom + offset[1]]; }
function imagePoint(e) { const b = canvas.getBoundingClientRect(); return [(e.clientX - b.left - offset[0]) / zoom, (e.clientY - b.top - offset[1]) / zoom]; }
function inside(p) { return p[0] >= 0 && p[1] >= 0 && p[0] < entry.width && p[1] < entry.height; }
function path(points, color, closed = false, dashed = false) { if (!points.length) return; ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [7, 5] : []); ctx.beginPath(); points.forEach((p, i) => { const s = screen(p); i ? ctx.lineTo(...s) : ctx.moveTo(...s); }); if (closed) ctx.closePath(); ctx.stroke(); ctx.setLineDash([]); }
function dot(p, text, color) { const [x, y] = screen(p); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke(); ctx.font = '12px system-ui'; ctx.lineWidth = 3; ctx.strokeStyle = '#111'; ctx.strokeText(text, x + 10, y - 8); ctx.fillStyle = color; ctx.fillText(text, x + 10, y - 8); }
function projection(H, p) { const d = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2]; return [(H[0][0] * p[0] + H[0][1] * p[1] + H[0][2]) / d, (H[1][0] * p[0] + H[1][1] * p[1] + H[1][2]) / d]; }
function inverse3(a) { const [[A, B, C], [D, E, F], [G, H, I]] = a; const c = [[E * I - F * H, C * H - B * I, B * F - C * E], [F * G - D * I, A * I - C * G, C * D - A * F], [D * H - E * G, B * G - A * H, A * E - B * D]]; const d = A * c[0][0] + B * c[1][0] + C * c[2][0]; return c.map(r => r.map(v => v / d)); }
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); if (!record || !image.width) return;
  ctx.drawImage(image, ...offset, image.width * zoom, image.height * zoom);
  const mode = $('mode').value; if (mode === 'shots') return;
  if (record.calibration && mode === 'calibration') {
    const H = inverse3(record.calibration.H);
    for (const x of [-89, -25, 0, 25, 89]) path([projection(H, [x, 42.5]), projection(H, [x, -42.5])], '#91e9ab', false, true);
    for (const [x, y] of [[0, 0], [-69, 22], [-69, -22], [69, 22], [69, -22]]) {
      const points = Array.from({ length: 65 }, (_, i) => projection(H, [x + 15 * Math.cos(i * Math.PI / 32), y + 15 * Math.sin(i * Math.PI / 32)])); path(points, '#91e9ab', false, true);
    }
    for (const end of [-1, 1]) for (const side of [-1, 1]) path([
      projection(H, [end * 89, side * 11]), projection(H, [end * 100, side * 14])
    ], '#91e9ab', false, true);
    path(record.calibration.supported_polygon, '#ff9f43', true, true);
    if (record.calibration.supported_polygon.length) { const [x, y] = screen(record.calibration.supported_polygon[0]); ctx.font = '12px system-ui'; ctx.lineWidth = 3; ctx.strokeStyle = '#111'; ctx.strokeText('fit support boundary — not rink paint', x + 8, y - 8); ctx.fillStyle = '#ff9f43'; ctx.fillText('fit support boundary — not rink paint', x + 8, y - 8); }
  }
  record.ignore_regions.forEach(p => path(p, '#ff9999', true, true));
  const feature = (f, color) => f.points.length === 1 ? dot(f.points[0], f.landmark || f.type, color) : path(f.points, color);
  if (mode === 'geometry' || mode === 'calibration') record.features.forEach((f, i) => feature(f, selection?.kind === 'features' && selection.index === i ? '#fff' : '#ffe568'));
  if (mode === 'geometry' && $('showProposals').checked) record.proposals.forEach(f => feature(f, '#ee9bff'));
  if (mode === 'players') record.players.forEach((p, i) => {
    const [x1, y1, x2, y2] = p.bbox; const color = selection?.kind === 'players' && selection.index === i ? '#fff' : '#68d8ff';
    path([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], color, true);
    if (p.ice_point) dot(p.ice_point, `${p.role} ${i + 1}`, color);
  }); path(draft, '#fff'); draft.forEach(p => dot(p, '', '#fff'));
  if (drag?.tool === 'box') path([drag.start, [drag.current[0], drag.start[1]], drag.current, [drag.start[0], drag.current[1]]], '#fff', true);
}
function drawRink() {
  const c = $('rink').getContext('2d'); c.clearRect(0, 0, 330, 165); const p = ([x, y]) => [165 + x * 1.5, 82.5 - y * 1.5];
  c.strokeStyle = '#71818a'; c.strokeRect(15, 18.75, 300, 127.5);
  for (const x of [-89, -25, 0, 25, 89]) { c.strokeStyle = Math.abs(x) === 25 ? '#2663c5' : '#c44545'; c.beginPath(); c.moveTo(...p([x, 42.5])); c.lineTo(...p([x, -42.5])); c.stroke(); }
  c.fillStyle = '#26343f'; c.font = '11px system-ui'; c.fillText('W / −x', 22, 12); c.fillText('N / +y', 142, 12); c.fillText('E / +x', 272, 12);
  for (const [name, point] of Object.entries(manifest.landmarks)) { c.fillStyle = name === $('landmark').value ? '#008560' : '#cc4a52'; c.beginPath(); c.arc(...p(point), name === $('landmark').value ? 5 : 3, 0, Math.PI * 2); c.fill(); }
}
function makeFeature(points) {
  return {
    id: uid(), type: $('tool').value === 'point' ? 'landmark' : $('featureType').value,
    landmark: $('tool').value === 'point' ? ($('landmark').value || null) : null, points, visibility: $('visibility').value, provenance: 'manual', held_out: $('heldOut').checked
  };
}
function addFeature(points) {
  if ($('tool').value === 'point' && !$('landmark').value) throw Error('Choose a named point on the rink before clicking the image.');
  if (['occluded', 'out_of_frame'].includes($('visibility').value)) throw Error('Use Record unseen for unseen features.');
  const f = makeFeature(points); if (points.length > 1 && f.type === 'landmark') throw Error('Choose a polyline feature type.');
  mutate(() => { record.features.push(f); selection = { kind: 'features', index: record.features.length - 1 }; record.review.geometry = 'in_progress'; });
}
function finish() {
  if (!draft.length) return; const tool = $('tool').value; if (tool === 'ignore') {
    if (draft.length < 3) throw Error('Ignore polygon needs three vertices.');
    const points = clone(draft); mutate(() => record.ignore_regions.push(points));
  } else { if (draft.length < 2) throw Error('Polyline needs two vertices.'); addFeature(clone(draft)); } draft = []; draw();
}
function startNewAnnotation() {
  selection = null; draft = []; $('visibility').value = 'visible'; $('heldOut').checked = false;
  if ($('mode').value === 'geometry') { $('landmark').value = ''; $('featureType').value = 'landmark'; setTool('point'); }
  else setTool('box');
}
canvas.onpointerdown = e => {
  if (!record || loading) return; const p = imagePoint(e), tool = $('tool').value;
  if (tool === 'inspect') { drag = { tool, client: [e.clientX, e.clientY], offset: [...offset] }; canvas.setPointerCapture(e.pointerId); return; }
  if (!inside(p)) return;
  try {
    if (tool === 'box') { drag = { tool, start: p, current: p }; canvas.setPointerCapture(e.pointerId); }
    else if (tool === 'point') addFeature([p]);
    else if (tool === 'line' || tool === 'ignore') { draft.push(p); draw(); }
    else if (tool === 'contact') {
      if (selection?.kind !== 'players') throw Error('Select a player first.');
      if ($('contactVisibility').value === 'hidden') throw Error('Choose the contact visibility before placing a point.');
      mutate(() => { selected().ice_point = p; selected().contact_visibility = $('contactVisibility').value; });
    } else if (tool === 'move') {
      if (selection?.kind === 'features' && selected().points.length) { const item = selected(); let i = 0; item.points.forEach((v, j) => { if (Math.hypot(v[0] - p[0], v[1] - p[1]) < Math.hypot(item.points[i][0] - p[0], item.points[i][1] - p[1])) i = j; }); mutate(() => item.points[i] = p); }
      else throw Error('Select an observed feature with points first.');
    }
  } catch (err) { message(err.message); }
};
canvas.onpointermove = e => { if (!drag) return; if (drag.tool === 'inspect') { offset = [drag.offset[0] + e.clientX - drag.client[0], drag.offset[1] + e.clientY - drag.client[1]]; } else { const p = imagePoint(e); drag.current = [Math.max(0, Math.min(entry.width - 1, p[0])), Math.max(0, Math.min(entry.height - 1, p[1]))]; } draw(); };
canvas.onpointerup = () => {
  if (drag?.tool === 'box') {
    const a = drag.start, b = drag.current; if (Math.abs(a[0] - b[0]) > 3 && Math.abs(a[1] - b[1]) > 3) mutate(() => {
      record.players.push({ id: uid(), role: $('role').value, team: $('role').value === 'official' ? 'official' : $('team').value, bbox: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])], ice_point: null, contact_visibility: $('contactVisibility').value, occluded: $('occluded').checked, truncated: $('truncated').checked, provenance: 'manual' });
      selection = { kind: 'players', index: record.players.length - 1 }; record.review.players = 'in_progress';
    });
  } drag = null; draw();
};
canvas.onpointercancel = () => { drag = null; draw(); };
canvas.onwheel = e => { e.preventDefault(); const p = imagePoint(e); const old = zoom; zoom = Math.min(8, Math.max(.08, zoom * Math.exp(-e.deltaY * .001))); offset = [offset[0] + p[0] * (old - zoom), offset[1] + p[1] * (old - zoom)]; draw(); };
$('rink').onclick = e => { const b = $('rink').getBoundingClientRect(); const x = (e.clientX - b.left) * 330 / b.width, y = (e.clientY - b.top) * 165 / b.height; let nearest = null, d = Infinity; for (const [name, [rx, ry]] of Object.entries(manifest.landmarks)) { const v = Math.hypot(x - (165 + rx * 1.5), y - (82.5 - ry * 1.5)); if (v < d) { d = v; nearest = name; } } selection = null; $('landmark').value = nearest; $('featureType').value = 'landmark'; $('visibility').value = 'visible'; setTool('point'); };
function run(fn) { return () => Promise.resolve().then(fn).catch(e => message(e.message)); }
async function init() {
  manifest = await api('/api/manifest'); $('mode').value = manifest.mode;
  options($('tool'), PHASES[manifest.mode].tools); for (const o of $('tool').options) o.textContent = TOOL_NAMES[o.value];
  options($('landmark'), ['', ...Object.keys(manifest.landmarks)]); options($('featureType'), manifest.feature_types); options($('visibility'), manifest.visibility); options($('geometryStatus'), manifest.tags.geometry_status);
  for (const o of $('landmark').options) o.textContent = landmarkName(o.value);
  for (const key of ['shot', 'landmarks', 'geometry', 'players']) { const label = document.createElement('label'); label.textContent = 'Completion for this phase'; const select = document.createElement('select'); select.id = 'review_' + key; options(select, ['unreviewed', 'in_progress', 'reviewed']); select.onchange = () => { const v = select.value; mutate(() => record.review[key] = v); }; label.append(select); $('reviews').append(label); }
  $('prev').onclick = run(() => navigate(-1)); $('next').onclick = run(() => navigate(1)); $('frame').onchange = run(() => load(Number($('frame').value)));
  $('search').oninput = frameMenu; $('unreviewed').onchange = frameMenu;
  $('geometryPilot').onchange = run(async () => { frameMenu(); const items = filtered(); if (items.length && !items.some(({ i }) => i === index)) await load(items[0].i); });
  $('mode').onchange = run(async () => {
    if (loading) return;
    loading = true;
    try {
      await save(); selection = null; draft = []; setTool(PHASES[$('mode').value].tools[0]); render(); frameMenu();
      await autoFit();
    } finally { loading = false; }
  });
  document.querySelectorAll('[data-phase]').forEach(b => b.onclick = run(async () => { await save(); $('mode').value = b.dataset.phase; $('mode').dispatchEvent(new Event('change')); }));
  $('tool').onchange = () => { draft = []; phaseUI(); draw(); }; $('showProposals').onchange = draw; $('fitView').onclick = fitView;
  $('newAnnotation').onclick = startNewAnnotation;
  $('traceLine').onclick = () => { selection = null; if ($('featureType').value === 'landmark') $('featureType').value = 'blue_line'; setTool('line'); };
  $('ignoreArea').onclick = $('ignorePlayers').onclick = () => { selection = null; setTool('ignore'); };
  $('placeContact').onclick = () => { setTool('contact'); if (selection?.kind !== 'players') message('Select a person in the list first.'); };
  $('completeNext').onclick = async () => {
    const button = $('completeNext'), feedback = $('completionFeedback');
    if (button.disabled) return;
    if (loading) { feedback.textContent = 'Please wait for the frame or rink fit to finish loading, then try again.'; return; }
    button.disabled = true; feedback.textContent = 'Saving review…';
    try {
      const phase = $('mode').value;
      await save();
      if (phase === 'calibration' && record.shot.geometry_status === 'accepted') {
        if (!record.calibration) throw Error('Review not completed. The automatic fit could not be calculated. Check the overlay error above and correct the landmarks, or choose another result.');
        if (record.orientation === 'unknown') throw Error('Review not completed. Set the rink axes above; the overlay will update automatically. Then select Accepted.');
      }
      if (phase === 'calibration' && !['accepted', 'insufficient', 'ambiguous', 'rejected'].includes(record.shot.geometry_status)) throw Error('Choose a fit result: accepted, insufficient, ambiguous, or rejected.');
      mutate(() => record.review[PHASES[phase].key] = 'reviewed'); await save(); frameMenu();
      const next = filtered().find(({ i }) => i > index);
      if (next) await load(next.i); else feedback.textContent = 'Review saved. No later frames match this filter. You can choose another frame or phase above.';
    } catch (e) { feedback.textContent = e.message; }
    finally { button.disabled = false; }
  };
  $('save').onclick = run(save); $('finish').onclick = run(finish);
  $('undo').onclick = () => { if (undoStack.length) { const current = record; record = undoStack.pop(); record.revision = current.revision; selection = null; record.calibration = current.calibration; if (JSON.stringify(record.features) !== JSON.stringify(current.features) || record.orientation !== current.orientation) { record.calibration = null; if (record.shot.geometry_status === 'accepted') record.shot.geometry_status = 'unreviewed'; record.review.geometry = 'in_progress'; } changed(); render(); } };
  $('delete').onclick = () => { if (selection) mutate(() => { record[selection.kind].splice(selection.index, 1); selection = null; }); };
  for (const [id, key] of [['shotId', 'shot_id'], ['leakageGroup', 'leakage_group'], ['notes', 'notes'], ['orientation', 'orientation']]) $(id).onchange = () => { const v = $(id).value; mutate(() => record[key] = v || (['shot_id', 'leakage_group'].includes(key) ? null : '')); };
  $('geometryStatus').onchange = () => {
    const v = $('geometryStatus').value;
    mutate(() => { record.shot.geometry_status = v; record.review.geometry = 'in_progress'; });
  };
  for (const [id, key] of [['landmark', 'landmark'], ['featureType', 'type'], ['visibility', 'visibility'], ['heldOut', 'held_out']]) $(id).onchange = () => {
    const v = id === 'heldOut' ? $(id).checked : $(id).value; if (selection?.kind === 'features') mutate(() => { selected()[key] = id === 'heldOut' ? v : (v || null); if (key === 'visibility' && ['occluded', 'out_of_frame'].includes(v)) selected().points = []; }); else { phaseUI(); drawRink(); }
  };
  for (const [id, key] of [['role', 'role'], ['team', 'team'], ['contactVisibility', 'contact_visibility'], ['occluded', 'occluded'], ['truncated', 'truncated']]) $(id).onchange = () => {
    const v = ['occluded', 'truncated'].includes(id) ? $(id).checked : $(id).value; if (selection?.kind === 'players') mutate(() => { selected()[key] = v; if (key === 'role') { if (v === 'official') selected().team = 'official'; else if (selected().team === 'official') selected().team = 'unknown'; } if (key === 'contact_visibility' && v === 'hidden') selected().ice_point = null; });
  };
  $('clearContact').onclick = () => { if (selection?.kind === 'players') mutate(() => { selected().ice_point = null; selected().contact_visibility = 'hidden'; }); };
  $('addUnseen').onclick = run(() => { if (!$('landmark').value || !['occluded', 'out_of_frame'].includes($('visibility').value)) throw Error('Choose a landmark and occluded/out-of-frame visibility.'); mutate(() => record.features.push({ id: uid(), type: 'landmark', landmark: $('landmark').value, points: [], visibility: $('visibility').value, provenance: 'manual', held_out: false })); });
  async function serverAction(path, extra = {}) { await save(); loading = true; try { record = await api(path, { image_id: record.image_id, revision: record.revision, ...extra }); undoStack = []; render(); } finally { loading = false; } }
  $('fitGeometry').onclick = run(async () => { await serverAction('/api/fit'); message('Fit is a proposal. Inspect the overlay and any available check-point errors before accepting.'); });
  $('proposeCenter').onclick = run(() => serverAction('/api/propose', { method: 'center' }));
  $('proposeGeneral').onclick = run(() => serverAction('/api/propose', { method: 'general' }));
  async function markUsability(value) {
    await save(); mutate(() => { record.shot.position_usability = value; record.review.shot = 'reviewed'; record.reason = ''; });
    await save(); frameMenu();
    const next = filtered().find(({ i }) => i > index);
    if (next) await load(next.i); else message('Saved. No later frames match this filter.');
  }
  $('markUsable').onclick = run(() => markUsability('usable'));
  $('markUnusable').onclick = run(() => markUsability('unusable'));
  window.addEventListener('beforeunload', e => { if (dirty || saving) { e.preventDefault(); e.returnValue = ''; } });
  window.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) || e.target.isContentEditable) return;
    const mode = $('mode').value, key = e.key.toLowerCase();
    if (mode === 'players' && key === 'backspace' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      if (e.repeat || loading || drag || $('completeNext').disabled || selection?.kind !== 'players') return;
      $('delete').click();
      return;
    }
    if (mode === 'players' && !e.ctrlKey && !e.metaKey && !e.altKey && /^[0-9]$/.test(key)) {
      e.preventDefault();
      if (e.repeat || loading || drag || $('completeNext').disabled) return;
      const personIndex = key === '0' ? 9 : Number(key) - 1;
      if (!record.players[personIndex]) return;
      selection = { kind: 'players', index: personIndex };
      render();
      $('annotations').querySelector('.item.selected')?.scrollIntoView({ block: 'nearest' });
      return;
    }
    if (mode === 'players' && !e.ctrlKey && !e.metaKey && !e.altKey && ['a', 'b', 'r', 's', 'g'].includes(key)) {
      e.preventDefault();
      if (e.repeat || loading || $('completeNext').disabled) return;
      const person = selection?.kind === 'players' ? selected() : null;
      const currentRole = person ? person.role : $('role').value;
      const currentTeam = person ? person.team : $('team').value;
      const role = { r: 'official', s: 'skater', g: 'goalie' }[key] || (['official', 'unknown'].includes(currentRole) ? 'skater' : currentRole);
      const team = { a: 'team_a', b: 'team_b', r: 'official' }[key] || (['team_a', 'team_b'].includes(currentTeam) ? currentTeam : 'unknown');
      if (person) mutate(() => { person.role = role; person.team = team; });
      else { $('role').value = role; $('team').value = team; }
      return;
    }
    if (mode === 'calibration' && !e.ctrlKey && !e.metaKey && !e.altKey && ['a', 'd', 'i', 'enter'].includes(key)) {
      e.preventDefault();
      if (e.repeat || loading || $('completeNext').disabled) return;
      if (key === 'enter') $('completeNext').click();
      else {
        $('geometryStatus').value = { a: 'accepted', d: 'rejected', i: 'insufficient' }[key];
        $('geometryStatus').dispatchEvent(new Event('change'));
        $('completionFeedback').textContent = `${{ a: 'Accepted', d: 'Rejected', i: 'Insufficient' }[key]} selected. Press Enter to save and continue.`;
      }
      return;
    }
    if (e.key === 'Escape') { draft = []; drag = null; draw(); }
    if (key === 'm' && mode === 'geometry') { e.preventDefault(); setTool('move'); }
    if (e.key === 'Enter') { e.preventDefault(); if (mode === 'geometry' && selection?.kind === 'features' && selected()?.type === 'landmark' && selected().points.length === 1) startNewAnnotation(); else run(finish)(); }
    if (e.key === 'ArrowRight') run(() => navigate(1))();
    if (e.key === 'ArrowLeft') run(() => navigate(-1))();
    if ((e.ctrlKey || e.metaKey) && key === 's') { e.preventDefault(); run(save)(); }
  });
  new ResizeObserver(resize).observe($('viewport')); if (manifest.images.length) { const last = manifest.images.findIndex(e => e.id === localStorage.getItem('ice-last-image')); const items = filtered(); const target = items.some(({ i }) => i === last) ? last : items[0]?.i; await load(target ?? 0); } else message('No images indexed. Run the index command first.');
}
init().catch(e => message(e.message));
