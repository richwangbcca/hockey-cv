// Exercise each labeling phase against the fixture in docs/LABELING_GUIDE.md.
// Chrome must expose a local debugging port; no npm dependencies are needed.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

const base = process.env.LABEL_TEST_URL || 'http://127.0.0.1:8766';
const chrome = process.env.CHROME_DEBUG_URL || 'http://127.0.0.1:9223';
const tabs = await (await fetch(chrome + '/json/list')).json();
const socket = new WebSocket(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
let sequence = 0; const pending = new Map(), exceptions = [];
socket.addEventListener('message', event => {
  const data = JSON.parse(event.data);
  if (data.method === 'Runtime.exceptionThrown') exceptions.push(data.params.exceptionDetails);
  if (data.id && pending.has(data.id)) { const { resolve, reject } = pending.get(data.id); pending.delete(data.id); data.error ? reject(Error(JSON.stringify(data.error))) : resolve(data.result); }
});
function cdp(method, params = {}) { return new Promise((resolve, reject) => { const id = ++sequence; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); }); }
async function evaluate(expression) { const result = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }); if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails)); return result.result.value; }
async function wait(expression) { for (let i = 0; i < 100; i++) { if (await evaluate(expression)) return; await new Promise(resolve => setTimeout(resolve, 100)); } throw Error('Timed out: ' + expression); }
async function set(id, value) { await evaluate(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change'));})()`); }
async function click(id) { await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`); }
async function pageLoad(method, params = {}) { const loaded = new Promise(resolve => { const listener = event => { if (JSON.parse(event.data).method === 'Page.loadEventFired') { socket.removeEventListener('message', listener); resolve(); } }; socket.addEventListener('message', listener); }); await cdp(method, params); await loaded; }
async function mouse(type, p) { await cdp('Input.dispatchMouseEvent', { type, x: p[0], y: p[1], button: type === 'mouseMoved' ? 'none' : 'left', clickCount: 1 }); }
async function point(p) { return evaluate(`(()=>{const b=canvas.getBoundingClientRect();const p=screen(${JSON.stringify(p)});return [b.left+p[0],b.top+p[1]];})()`); }
try {
  await cdp('Runtime.enable'); await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1050, deviceScaleFactor: 1, mobile: false });
  await pageLoad('Page.navigate', { url: base });
  await wait(`typeof record!=='undefined' && !!record && !loading`);
  assert.equal(await evaluate('manifest.images.length'), 1);
  assert.equal(await evaluate('record.proposals.length'), 8, 'Use a fresh disposable fixture');
  assert.equal(await evaluate(`document.getElementById('geometryPanel').hidden`), true);
  assert.equal(await evaluate(`document.getElementById('suggestionPanel').hidden`), true);
  assert.deepEqual(await evaluate(`Array.from(document.getElementById('tool').options,o=>o.value)`), ['inspect']);
  const shotScreenshot = await cdp('Page.captureScreenshot', { format: 'png' });
  await fs.writeFile('/tmp/homography-phase-shots.png', Buffer.from(shotScreenshot.data, 'base64'));
  await set('shotId', 'test-shot'); await click('markUsable'); await wait('!dirty && !saving && record.review.shot === "reviewed"');
  assert.equal(await evaluate('record.shot.position_usability'), 'usable');
  assert.equal(await evaluate('record.review.landmarks'), 'unreviewed');
  await set('mode', 'geometry');
  assert.equal(await evaluate(`document.getElementById('calibrationPanel').hidden`), true);
  assert.equal(await evaluate(`document.getElementById('playerPanel').hidden`), true);
  assert.equal(await evaluate(`document.getElementById('landmark').options[1].textContent`), 'Center-ice dot');
  await set('tool', 'move'); await click('newAnnotation');
  assert.equal(await evaluate(`document.getElementById('tool').value`), 'point');
  assert.equal(await evaluate(`document.getElementById('featureType').value`), 'landmark');
  for (let i = 0; i < 8; i++)await evaluate(`document.querySelector('#proposals button').click()`);
  await set('tool', 'inspect');
  await evaluate(`window.dispatchEvent(new KeyboardEvent('keydown',{key:'m',bubbles:true}))`);
  assert.equal(await evaluate(`document.getElementById('tool').value`), 'move');
  await evaluate(`window.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))`);
  assert.equal(await evaluate('selection'), null);
  assert.equal(await evaluate(`document.getElementById('tool').value`), 'point');
  assert.equal(await evaluate(`document.getElementById('landmark').value`), '');
  await click('completeNext'); await wait('!dirty && !saving && record.review.landmarks === "reviewed"');
  assert.equal(await evaluate('record.review.geometry'), 'in_progress');
  await set('mode', 'calibration');
  assert.equal(await evaluate(`document.getElementById('geometryPanel').hidden`), true);
  assert.equal(await evaluate(`document.getElementById('calibrationPanel').hidden`), false);
  await set('orientation', 'sequence_local');
  await evaluate(`document.querySelectorAll('#annotations .item')[0].click()`);
  await evaluate(`document.getElementById('heldOut').checked=true;document.getElementById('heldOut').dispatchEvent(new Event('change'))`);
  await click('save'); await wait('!dirty && !saving');
  await click('fitGeometry'); await wait('!!record.calibration && !loading');
  assert.equal(await evaluate('record.calibration.validation_count'), 1);
  assert.equal(await evaluate('record.review.landmarks'), 'reviewed');
  await set('geometryStatus', 'accepted'); await click('completeNext'); await wait('!dirty && !saving && record.review.geometry === "reviewed"');
  await set('mode', 'players'); await set('tool', 'box');
  const a = await point([1180, 650]), b = await point([1340, 850]);
  await mouse('mousePressed', a); await mouse('mouseMoved', b); await mouse('mouseReleased', b);
  assert.equal(await evaluate('record.players.length'), 1);
  await set('contactVisibility', 'both'); await set('tool', 'contact');
  const contact = await point([1280, 830]); await mouse('mousePressed', contact); await mouse('mouseReleased', contact);
  assert.ok(await evaluate('record.players[0].ice_point !== null'));
  await click('completeNext'); await wait('!dirty && !saving && record.review.players === "reviewed"');
  const screenshot = await cdp('Page.captureScreenshot', { format: 'png' });
  await fs.writeFile('/tmp/homography-labeling-browser.png', Buffer.from(screenshot.data, 'base64'));
  await pageLoad('Page.reload'); await wait(`typeof record!=='undefined' && !!record && !loading`);
  assert.equal(await evaluate('record.players.length'), 1);
  assert.equal(await evaluate('record.shot.geometry_status'), 'accepted');
  // Editing an observed anchor must invalidate the accepted fit immediately and on disk.
  await set('mode', 'geometry'); await evaluate(`document.querySelector('#annotations .item').click()`);
  await set('tool', 'move'); const moved = await point([800, 590]); await mouse('mousePressed', moved); await mouse('mouseReleased', moved);
  assert.equal(await evaluate('record.calibration'), null);
  await click('save'); await wait('!dirty && !saving');
  const stored = await (await fetch(base + '/api/record/' + await evaluate('record.image_id'))).json();
  assert.equal(stored.calibration, null); assert.equal(stored.shot.geometry_status, 'unreviewed');
  assert.equal(stored.review.landmarks, 'in_progress');
  assert.equal(exceptions.length, 0, JSON.stringify(exceptions));
  console.log('PASS: shot tags, proposal review, held-out calibration, acceptance, player box/contact, persistence, geometry invalidation; no browser exceptions.');
} finally { socket.close(); }
