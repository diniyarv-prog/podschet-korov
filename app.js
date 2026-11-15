import { CowTracker } from './tracker.js';
import { CrossingCounter } from './counter.js';
import { directionVectors } from './geometry.js';
import { MotionGuard } from './motion-guard.js';
import { EvidenceBuffer } from './evidence.js';
import { createSession, addAutomaticEvent, addManualEvent, reviewEvent, totals } from './session.js';
import { SessionStore } from './session-store.js';

const get = (id) => document.getElementById(id);
const video = get('video');
const canvas = get('overlay');
const context = canvas.getContext('2d');
const status = get('status');
const debug = get('debug');
const endpointButtons = {start: get('line-start'), end: get('line-end')};
const view = get('view');
const directionSelect = get('direction');
const start = get('start');
const tracker = new CowTracker();
const counter = new CrossingCounter();
const motionGuard = new MotionGuard();
const evidence = new EvidenceBuffer();
const sessionStore = new SessionStore();
const inputFrame = document.createElement('canvas');
const inputContext = inputFrame.getContext('2d');
const motionFrame = document.createElement('canvas');
motionFrame.width = 64;
motionFrame.height = 36;
const motionContext = motionFrame.getContext('2d', {willReadFrequently: true});
const presets = {
  side: {direction: 'right', hint: 'Держи ворота на одном месте в кадре.'},
  overhead: {direction: 'down', hint: 'Держи в кадре участок, через который проходят коровы.'}
};
const directions = {
  right: {axis: 0},
  left: {axis: 0},
  down: {axis: 1},
  up: {axis: 1}
};
let currentView = view.value;
let modelPromise;
let stream;
let fileUrl;
let videoFilename = null;
let source = 0;
let mediaHandlers = {};
let cameraTrackHandlers = [];
let cameraRecoveryRequired = false;
let run = 0;
let counting = false;
let ready = false;
let tracks = [];
let currentSession = null;
let runStartedAt = 0;
let previousMotionFrame = null;
let inferenceRunning = false;
let detectionFrame;
let lastVideoTime = null;
let processedTimes = [];
// Storage writes, including quota retries, must finish in review-action order.
let saveQueue = Promise.resolve();
let countingLine = {start: [0.5, 0.15], end: [0.5, 0.85]};
let draggedEndpoint = null;
let draggedPointerId = null;
let lineAdjusted = false;
// Populated by the detection pipeline; unavailable measurements stay blank.
const diagnostics = {inferenceMs: null, processedFps: null, motionBlocked: false};

function script(url) {
  return new Promise((resolve, reject) => {
    const item = document.createElement('script');
    item.src = url;
    item.onload = resolve;
    item.onerror = reject;
    document.head.append(item);
  });
}

function loadModel() {
  modelPromise ??= (async () => {
    if (!window.tf) await script('./vendor/tf.min.js');
    if (!window.cocoSsd) await script('./vendor/coco-ssd.min.js');
    await window.tf.ready();
    const config = {base: 'lite_mobilenet_v2', modelUrl: './vendor/model/model.json'};
    try {
      return await window.cocoSsd.load(config);
    } catch (error) {
      if (window.tf.getBackend() !== 'webgl') throw error;
      await window.tf.setBackend('cpu');
      return window.cocoSsd.load(config);
    }
  })().catch(error => {
    modelPromise = undefined;
    throw error;
  });
  return modelPromise;
}

function cameraInstructions() {
  return 'Поставь линию поперёк прохода. ' + presets[currentView].hint;
}

function setCounting(active) {
  if (counting && !active) lineAdjusted = false;
  counting = active;
  start.textContent = active ? 'Завершить подсчёт' : 'Начать новый подсчёт';
  finishDrag();
  view.disabled = active;
  directionSelect.disabled = active;
  redraw();
}

function frameRect() {
  const {width, height} = canvas.getBoundingClientRect();
  if (!video.videoWidth || !video.videoHeight) return {left: 0, top: 0, width: 0, height: 0};
  const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
  const frameW = video.videoWidth * scale;
  const frameH = video.videoHeight * scale;
  return {left: (width - frameW) / 2, top: (height - frameH) / 2, width: frameW, height: frameH};
}

function eventPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  const frame = frameRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - bounds.left - frame.left) / frame.width)),
    Math.max(0, Math.min(1, (event.clientY - bounds.top - frame.top) / frame.height))
  ];
}

function endpointDistance(point, endpoint) {
  const frame = frameRect();
  return Math.hypot((point[0] - endpoint[0]) * frame.width, (point[1] - endpoint[1]) * frame.height);
}

function canvasPoint([x, y], frame) {
  return [frame.left + x * frame.width, frame.top + y * frame.height];
}

function useDefaultLine() {
  if (lineAdjusted) return;
  countingLine = directions[directionSelect.value].axis === 0
    ? {start: [0.5, 0.15], end: [0.5, 0.85]}
    : {start: [0.15, 0.5], end: [0.85, 0.5]};
}

function moveEndpoint(endpoint, point) {
  const other = countingLine[endpoint === 'start' ? 'end' : 'start'];
  // Keep a valid segment for the counter, including after keyboard adjustment.
  if (Math.hypot(point[0] - other[0], point[1] - other[1]) < 0.01) return;
  countingLine[endpoint] = point;
  lineAdjusted = true;
  redraw();
}

function beginDrag(event, endpoint) {
  if (counting || !video.videoWidth || draggedEndpoint || event.isPrimary === false || event.button !== 0) return;
  const frame = frameRect();
  if (!frame.width || !frame.height) return;
  const point = eventPoint(event);
  const nearest = endpoint ?? (endpointDistance(point, countingLine.start) <= endpointDistance(point, countingLine.end) ? 'start' : 'end');
  // Use the raw position for hit testing: a distant letterbox tap is not a handle tap.
  const bounds = canvas.getBoundingClientRect();
  const [x, y] = canvasPoint(countingLine[nearest], frame);
  if (!endpoint && Math.hypot(event.clientX - bounds.left - x, event.clientY - bounds.top - y) > 36) return;
  draggedEndpoint = nearest;
  draggedPointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function finishDrag(event) {
  if (event && event.pointerId !== draggedPointerId) return;
  const pointerId = draggedPointerId;
  draggedEndpoint = null;
  draggedPointerId = null;
  if (pointerId !== null && canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
}

canvas.addEventListener('pointerdown', event => beginDrag(event));
canvas.addEventListener('pointermove', event => {
  if (!draggedEndpoint || event.pointerId !== draggedPointerId || counting) return;
  moveEndpoint(draggedEndpoint, eventPoint(event));
});
canvas.addEventListener('pointerup', finishDrag);
canvas.addEventListener('pointercancel', finishDrag);
canvas.addEventListener('lostpointercapture', finishDrag);
for (const [endpoint, button] of Object.entries(endpointButtons)) {
  button.addEventListener('pointerdown', event => beginDrag(event, endpoint));
  button.addEventListener('keydown', event => {
    const delta = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.key];
    if (!delta || counting || !video.videoWidth) return;
    event.preventDefault();
    const step = event.shiftKey ? 0.05 : 0.01;
    moveEndpoint(endpoint, countingLine[endpoint].map((value, index) => Math.max(0, Math.min(1, value + delta[index] * step))));
  });
}

function drawCountingLine(frame) {
  const [sx, sy] = canvasPoint(countingLine.start, frame);
  const [ex, ey] = canvasPoint(countingLine.end, frame);
  const dx = countingLine.end[0] - countingLine.start[0];
  const dy = countingLine.end[1] - countingLine.start[1];
  const length = Math.hypot(dx, dy) || 1;
  let nx = -dy / length;
  let ny = dx / length;
  const [tx, ty] = directionVectors[directionSelect.value];
  if (nx * tx + ny * ty < 0) { nx *= -1; ny *= -1; }
  // Match the counter's normalized distances, including non-square video.
  const entryX = nx * counter.entryDistance * frame.width;
  const entryY = ny * counter.entryDistance * frame.height;
  const exitX = nx * counter.exitDistance * frame.width;
  const exitY = ny * counter.exitDistance * frame.height;
  context.fillStyle = 'rgba(255, 215, 128, 0.12)';
  context.beginPath();
  context.moveTo(sx - entryX, sy - entryY);
  context.lineTo(ex - entryX, ey - entryY);
  context.lineTo(ex + exitX, ey + exitY);
  context.lineTo(sx + exitX, sy + exitY);
  context.closePath();
  context.fill();
  context.strokeStyle = '#ffd780';
  context.lineWidth = 3;
  context.setLineDash([8, 5]);
  context.beginPath();
  context.moveTo(sx, sy);
  context.lineTo(ex, ey);
  context.stroke();
  context.setLineDash([]);
}

function drawHandles(frame) {
  for (const [endpoint, button] of Object.entries(endpointButtons)) {
    button.hidden = counting || !frame.width || !frame.height;
    if (button.hidden) continue;
    const [x, y] = canvasPoint(countingLine[endpoint], frame);
    const bounds = canvas.getBoundingClientRect();
    // Keep the full touch target available even when the endpoint reaches an edge.
    button.style.left = `${Math.max(24, Math.min(bounds.width - 24, x))}px`;
    button.style.top = `${Math.max(24, Math.min(bounds.height - 24, y))}px`;
    context.strokeStyle = '#ffd780';
    context.fillStyle = '#14221d';
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, 21, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = '#ffd780';
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fill();
  }
}

function renderDiagnostics() {
  const panel = get('diagnostics');
  panel.hidden = !debug.checked;
  panel.dataset.motionBlocked = String(diagnostics.motionBlocked);
  status.dataset.motionBlocked = String(diagnostics.motionBlocked);
  get('inference-ms').textContent = Number.isFinite(diagnostics.inferenceMs) ? `${Math.round(diagnostics.inferenceMs)} мс` : '—';
  get('processed-fps').textContent = Number.isFinite(diagnostics.processedFps) ? `${diagnostics.processedFps.toFixed(1)} кадр/с` : '—';
  get('motion-state').textContent = diagnostics.motionBlocked ? 'пауза: тряска' : 'стабильно';
}

function redraw() {
  renderDiagnostics();
  const {width, height} = canvas.getBoundingClientRect();
  const frame = frameRect();
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!frame.width || !frame.height) {
    for (const button of Object.values(endpointButtons)) button.hidden = true;
    return;
  }
  context.save();
  context.beginPath();
  context.rect(frame.left, frame.top, frame.width, frame.height);
  context.clip();
  drawCountingLine(frame);
  for (const {box, counted, history = []} of tracks) {
    const [x, y, w, h] = box;
    const [left, top] = canvasPoint([x, y], frame);
    context.lineWidth = 2;
    context.strokeStyle = counted ? '#b4f492' : '#82e4ff';
    context.strokeRect(left, top, w * frame.width, h * frame.height);
    if (debug.checked && history.length) {
      context.beginPath();
      history.map(item => canvasPoint(item.point, frame)).forEach(([px, py], index) => {
        if (index === 0) context.moveTo(px, py);
        else context.lineTo(px, py);
      });
      context.stroke();
    }
  }
  for (const {box, id, counted} of tracks) {
    const [left, top] = canvasPoint(box, frame);
    const label = counted ? `#${id} ✓` : `#${id}`;
    context.font = 'bold 15px system-ui';
    const labelWidth = context.measureText(label).width + 8;
    const labelX = Math.max(frame.left, Math.min(left, frame.left + frame.width - labelWidth));
    const labelY = Math.max(frame.top, Math.min(top, frame.top + frame.height - 24));
    context.fillStyle = '#14221d';
    context.fillRect(labelX, labelY, labelWidth, 24);
    context.strokeStyle = counted ? '#b4f492' : '#82e4ff';
    context.fillStyle = context.strokeStyle;
    context.fillText(label, labelX + 4, labelY + 17);
  }
  context.restore();
  drawHandles(frame);
}

function bindMediaHandlers(id) {
  for (const [type, handler] of Object.entries(mediaHandlers)) video.removeEventListener(type, handler);
  mediaHandlers = {
    ended() {
      if (id !== source || !fileUrl || !video.ended) return;
      finishCounting('Видео закончилось. Проверь события и итог.');
    },
    error() {
      if (id !== source || !video.error) return;
      finishCounting('Видео остановилось с ошибкой. Результат доступен для проверки.');
      ready = false;
      start.disabled = true;
    },
  };
  for (const [type, handler] of Object.entries(mediaHandlers)) video.addEventListener(type, handler);
}

function clearCameraTrackHandlers() {
  for (const {track, handler} of cameraTrackHandlers) track.removeEventListener('ended', handler);
  cameraTrackHandlers = [];
}

function stopForCameraTrackEnd(id, cameraStream) {
  if (id !== source || stream !== cameraStream) return;
  const message = 'Камера перестала передавать видео. Открой камеру снова и проверь результат.';
  if (counting) finishCounting(message);
  else {
    // The source is no longer usable even if an inference began during preview.
    run++;
    video.pause();
    status.textContent = message;
    renderReview();
    redraw();
  }
  cameraRecoveryRequired = true;
  ready = false;
  start.disabled = true;
}

function bindCameraTrackHandlers(id, cameraStream) {
  clearCameraTrackHandlers();
  for (const track of cameraStream.getVideoTracks?.() ?? cameraStream.getTracks()) {
    const handler = () => stopForCameraTrackEnd(id, cameraStream);
    track.addEventListener('ended', handler);
    cameraTrackHandlers.push({track, handler});
  }
}

function resetSource() {
  // Stopping a source is intentional; detach before stopping its tracks.
  clearCameraTrackHandlers();
  finishCounting();
  source++;
  bindMediaHandlers(source);
  cancelAnimationFrame(detectionFrame);
  ready = false;
  start.disabled = true;
  setCounting(false);
  lineAdjusted = false;
  resetProcessing();
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  get('camera').textContent = 'Включить камеру';
  video.pause();
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  if (fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl = undefined;
  videoFilename = null;
  get('placeholder').hidden = false;
  redraw();
  return source;
}

async function prepare(id) {
  get('placeholder').hidden = true;
  redraw();
  status.textContent = 'Загружаю модель распознавания…';
  let model;
  try {
    model = await loadModel();
  } catch {
    if (id === source) status.textContent = 'Модель не загрузилась. Проверь интернет и открой камеру или видео снова.';
    return;
  }
  // A dead camera can finish loading the model after its track ended. Do not
  // revive its controls; only opening a new source clears this recovery gate.
  if (id !== source || (cameraRecoveryRequired && !fileUrl)) return;
  ready = true;
  start.disabled = false;
  start.textContent = 'Начать подсчёт';
  status.textContent = fileUrl
    ? 'Поставь линию поперёк прохода, выбери направление и начни подсчёт.'
    : cameraInstructions();
  if (fileUrl) {
    video.pause();
  }
  scheduleDetection(id, model);
}

function resetProcessing() {
  tracker.reset();
  counter.reset();
  motionGuard.reset();
  evidence.reset();
  previousMotionFrame = null;
  lastVideoTime = null;
  processedTimes = [];
  tracks = [];
  Object.assign(diagnostics, {inferenceMs: null, processedFps: null, motionBlocked: false});
  get('total').value = '0';
  // Do not release inferenceRunning here: an obsolete model call still owns
  // inputFrame until its promise settles, even after the source/run changes.
}

function elapsedTime() {
  return Math.max(0, fileUrl ? video.currentTime * 1000 : performance.now() - runStartedAt);
}

function drawFreshInputFrame() {
  const ratio = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.round(video.videoWidth * ratio);
  const height = Math.round(video.videoHeight * ratio);
  if (!width || !height) return false;
  if (inputFrame.width !== width || inputFrame.height !== height) {
    inputFrame.width = width;
    inputFrame.height = height;
  }
  inputContext.drawImage(video, 0, 0, width, height);
  return true;
}

async function inferFrame(model, id, generation) {
  const started = performance.now();
  const predictions = await model.detect(inputFrame, 40, 0.18);
  const finished = performance.now();
  if (id === source && generation === run && !document.hidden) {
    processedTimes.push(finished);
    processedTimes = processedTimes.filter(time => finished - time <= 2000);
    diagnostics.inferenceMs = finished - started;
    diagnostics.processedFps = processedTimes.length / 2;
  }
  return predictions;
}

function scheduleDetection(id, model) {
  detectionFrame = requestAnimationFrame(async () => {
    if (id !== source || !ready) return;
    if (document.hidden || video.readyState < 2 || video.paused || video.seeking ||
        inferenceRunning || video.currentTime === lastVideoTime) {
      scheduleDetection(id, model);
      return;
    }
    const generation = run;
    inferenceRunning = true;
    try {
      if (drawFreshInputFrame()) {
        lastVideoTime = video.currentTime;
        const frameTime = counting ? elapsedTime() : performance.now();
        const predictions = await inferFrame(model, id, generation);
        if (id === source && generation === run && !document.hidden) {
          processPredictions(id, predictions, frameTime);
        }
      }
    } catch {
      if (id === source && generation === run) stopWithRecognitionError();
    } finally {
      inferenceRunning = false;
    }
    if (id === source && ready) scheduleDetection(id, model);
  });
}

function grayscaleSample() {
  // Use the captured inference frame, not the video that advanced while the
  // model was running, so motion, boxes and evidence describe the same moment.
  motionContext.drawImage(inputFrame, 0, 0, motionFrame.width, motionFrame.height);
  const rgba = motionContext.getImageData(0, 0, motionFrame.width, motionFrame.height).data;
  const gray = new Uint8Array(motionFrame.width * motionFrame.height);
  for (let pixel = 0, offset = 0; pixel < gray.length; pixel += 1, offset += 4) {
    gray[pixel] = Math.round(rgba[offset] * 0.299 + rgba[offset + 1] * 0.587 + rgba[offset + 2] * 0.114);
  }
  return gray;
}

function processPredictions(id, predictions, frameTime) {
  if (id !== source) return;
  const {width, height} = inputFrame;
  const detections = predictions.filter(({class: label}) => label === 'cow')
    .map(({bbox: [x, y, w, h], score}) => ({bbox: [x / width, y / height, w / width, h / height], score}));
  tracks = tracker.update(detections, frameTime);
  const sample = grayscaleSample();
  const motion = previousMotionFrame
    ? motionGuard.update(previousMotionFrame, sample, motionFrame.width, motionFrame.height, tracks.map(track => track.box))
    : {blocked: false, dx: 0, dy: 0, error: 0};
  previousMotionFrame = sample;
  diagnostics.motionBlocked = motion.blocked;
  const events = counter.update(tracks, frameTime, countingLine, directionSelect.value, {
    counting, motionBlocked: motion.blocked,
  });
  if (counting && currentSession) {
    const jpeg = inputFrame.toDataURL('image/jpeg', 0.7);
    // Complete only captures from earlier inferences before opening new ones.
    for (const eventId of evidence.pending.keys()) evidence.complete(eventId, jpeg);
    for (const event of events) addAutomaticEvent(currentSession, event, evidence.capture(event.id, jpeg));
    evidence.push(jpeg);
    currentSession.totals = totals(currentSession);
    const message = motion.blocked
      ? 'Камера движется. Автоматический подсчёт на паузе — держи камеру неподвижно.'
      : 'Идёт подсчёт. Рамка становится зелёной после засчитанного прохода.';
    if (status.textContent !== message) status.textContent = message;
  }
  tracks = tracks.map(track => ({...track, counted: counter.states.get(track.id)?.counted ?? false}));
  get('total').value = String(counter.total);
  redraw();
}

function saveSession(session) {
  session.totals = totals(session);
  const snapshot = structuredClone(session);
  saveQueue = saveQueue.then(() => sessionStore.save(snapshot)).then(result => {
    if (currentSession !== session) return;
    const message = get('storage-message');
    message.textContent = result.savedEvidence ? ''
      : 'Не хватило памяти для контрольных кадров. Итог и события сохранены без изображений.';
    message.hidden = !message.textContent;
  }).catch(() => {
    if (currentSession !== session) return;
    const message = get('storage-message');
    message.textContent = 'Не удалось сохранить результат на устройстве. Он доступен здесь, пока страница открыта.';
    message.hidden = false;
  });
}

function finishCounting(message) {
  if (!counting) return;
  // Close the event gate and invalidate awaited work before pausing/replacing media.
  setCounting(false);
  run++;
  if (currentSession && currentSession.endedAt === null) {
    currentSession.endedAt = Date.now();
    currentSession.durationMs = elapsedTime();
    saveSession(currentSession);
  }
  evidence.reset();
  video.pause();
  diagnostics.motionBlocked = false;
  if (message) status.textContent = message;
  renderReview();
  redraw();
}

function stopWithRecognitionError() {
  finishCounting();
  ready = false;
  start.disabled = true;
  video.pause();
  status.textContent = 'Распознавание остановилось. Открой камеру или видео снова. Результат доступен для проверки.';
}

function formatTime(milliseconds) {
  const tenths = Math.max(0, Math.round(milliseconds / 100));
  const seconds = Math.floor(tenths / 10);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')},${tenths % 10}`;
}

const eventReasons = {
  'confirmed-crossing': 'Проход через линию',
  'camera-motion': 'Движение камеры — проверь проход',
  'low-confidence': 'Низкая уверенность распознавания',
  'track-gap': 'Цель пропадала между кадрами',
};
const reviewLabels = {pending: 'Нужна проверка', accepted: 'Учтён', confirmed: 'Подтверждён', removed: 'Удалён'};

function renderReview() {
  get('review').hidden = !currentSession || counting;
  if (!currentSession) return;
  const session = currentSession;
  session.totals = totals(session);
  get('automatic-total').value = String(session.totals.automatic);
  get('confirmed-total').value = String(session.totals.confirmed);
  const list = get('review-events');
  list.replaceChildren();
  const priority = event => event.review === 'pending' ? 0 : event.kind === 'uncertain' ? 1 : 2;
  const events = [...session.events].sort((a, b) => priority(a) - priority(b) || a.time - b.time);
  if (!events.length) {
    const empty = document.createElement('p');
    empty.textContent = 'Проходов пока нет. Если корова прошла незамеченной, добавь проход вручную.';
    list.append(empty);
  }
  for (const event of events) {
    const row = document.createElement('article');
    row.className = 'review-event';
    row.dataset.review = event.review;
    const description = document.createElement('div');
    description.className = 'event-description';
    const time = document.createElement('time');
    time.dateTime = `PT${event.time / 1000}S`;
    time.textContent = formatTime(event.time);
    const reason = document.createElement('p');
    reason.textContent = event.source === 'manual'
      ? event.reason || 'Добавлено вручную'
      : eventReasons[event.reason] ?? 'Проход требует проверки';
    const confidence = document.createElement('p');
    if (Number.isFinite(event.confidence)) {
      confidence.className = 'event-confidence';
      confidence.textContent = `Уверенность: ${Math.round(event.confidence * 100)}%`;
    }
    const state = document.createElement('span');
    state.className = 'event-review-state';
    state.textContent = reviewLabels[event.review];
    description.append(time, reason);
    if (confidence.textContent) description.append(confidence);
    description.append(state);
    const pictures = document.createElement('div');
    pictures.className = 'event-pictures';
    for (const [slot, label] of [['before', 'До'], ['during', 'Проход'], ['after', 'После']]) {
      if (!event.evidence?.[slot]) continue;
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.src = event.evidence[slot];
      image.alt = `${label}: ${formatTime(event.time)}`;
      image.loading = 'lazy';
      const caption = document.createElement('figcaption');
      caption.textContent = label;
      figure.append(image, caption);
      pictures.append(figure);
    }
    const evidenceDetails = document.createElement('details');
    evidenceDetails.className = 'event-evidence';
    evidenceDetails.open = event.review === 'pending';
    const evidenceSummary = document.createElement('summary');
    evidenceSummary.textContent = 'Контрольные кадры';
    evidenceDetails.append(evidenceSummary, pictures);
    const actions = document.createElement('div');
    actions.className = 'event-actions';
    for (const [review, label] of [['confirmed', 'Подтвердить'], ['removed', 'Удалить']]) {
      const button = document.createElement('button');
      button.textContent = label;
      button.disabled = event.review === review;
      button.setAttribute('aria-label', `${label} проход в ${formatTime(event.time)}`);
      button.addEventListener('click', () => {
        if (currentSession !== session || counting) return;
        reviewEvent(session, event.id, review);
        renderReview();
        saveSession(session);
      });
      actions.append(button);
    }
    row.append(description, evidenceDetails, actions);
    list.append(row);
  }
}

function parseVideoTime(value) {
  const parts = value.trim().replace(',', '.').split(':');
  if (!parts.length || parts.length > 3 || parts.some(part => !/^\d+(?:\.\d+)?$/.test(part))) return NaN;
  if (parts.length > 1 && (Number(parts.at(-1)) >= 60 || parts.slice(0, -1).some(part => !/^\d+$/.test(part)))) return NaN;
  if (parts.length === 3 && Number(parts[1]) >= 60) return NaN;
  return parts.reduce((time, part) => time * 60 + Number(part), 0) * 1000;
}

async function resumeCamera(id) {
  await video.play();
  if (id !== source) return;
  get('camera').textContent = 'Перезапустить камеру';
  prepare(id);
}

async function openCamera() {
  if (stream?.active && !ready && video.paused && !cameraRecoveryRequired) {
    const id = source;
    try {
      await resumeCamera(id);
    } catch {
      if (id === source) status.textContent = 'Safari не запустил видео. Обнови страницу или выбери снятое видео.';
    }
    return;
  }
  const id = resetSource();
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = 'Камера доступна по HTTPS. Открой страницу в Safari или выбери снятое видео.';
    return;
  }
  status.textContent = 'Запрашиваю доступ к задней камере…';
  try {
    const nextStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: {ideal: 'environment'}, width: {ideal: 1280}, height: {ideal: 720} }
    });
    if (id !== source) {
      nextStream.getTracks().forEach(track => track.stop());
      return;
    }
    stream = nextStream;
    cameraRecoveryRequired = false;
    bindCameraTrackHandlers(id, stream);
    video.srcObject = stream;
    try {
      await resumeCamera(id);
    } catch {
      if (id === source) {
        get('camera').textContent = 'Включить камеру';
        status.textContent = 'Камера разрешена. Нажми «Включить камеру», чтобы запустить видео.';
      }
    }
  } catch (error) {
    if (id !== source) return;
    stream?.getTracks().forEach(track => track.stop());
    stream = undefined;
    status.textContent = error.name === 'NotAllowedError'
      ? 'Разреши камеру в настройках этого сайта в Safari, затем нажми «Включить камеру». Можно выбрать снятое видео.'
      : 'Камера не открылась. Нажми «Включить камеру» ещё раз или выбери снятое видео.';
  }
}

start.addEventListener('click', async () => {
  if (!ready) return;
  if (counting) {
    finishCounting('Подсчёт завершён. Проверь события и итог.');
    return;
  }
  run++;
  const generation = run;
  const id = source;
  resetProcessing();
  if (fileUrl) video.currentTime = 0;
  runStartedAt = performance.now();
  currentSession = createSession({
    view: currentView,
    direction: directionSelect.value,
    line: structuredClone(countingLine),
    sourceType: fileUrl ? 'file' : 'camera',
    videoFilename,
    videoDurationMs: fileUrl && Number.isFinite(video.duration) ? video.duration * 1000 : null,
  });
  get('storage-message').hidden = true;
  get('storage-message').textContent = '';
  get('review-message').textContent = '';
  setCounting(true);
  renderReview();
  status.textContent = 'Идёт подсчёт. Рамка становится зелёной после засчитанного прохода.';
  if (fileUrl || video.paused) {
    try {
      await video.play();
      if (id === source && generation !== run && !counting) video.pause();
    } catch {
      if (id === source && generation === run) finishCounting('Видео не запустилось. Попробуй открыть его снова.');
    }
  }
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) return;
  finishCounting('Подсчёт завершён: экран был скрыт. Проверь события перед новым запуском.');
  video.pause();
});
window.addEventListener('pagehide', () => {
  finishCounting();
  video.pause();
});
get('add-missed').addEventListener('click', () => {
  if (!currentSession || counting) return;
  let time = currentSession.durationMs ?? 0;
  if (currentSession.settings.sourceType === 'file') {
    const answer = window.prompt('Время прохода в видео: секунды или ММ:СС', formatTime(time));
    if (answer === null) return;
    time = parseVideoTime(answer);
    const duration = currentSession.settings.videoDurationMs;
    if (!Number.isFinite(time) || time < 0 || (Number.isFinite(duration) && time > duration)) {
      get('review-message').textContent = 'Укажи время в пределах видео: секунды или ММ:СС.';
      return;
    }
  }
  const note = window.prompt('Комментарий к добавленному проходу', 'Добавлено вручную');
  if (note === null) return;
  get('review-message').textContent = '';
  addManualEvent(currentSession, time, note.trim() || 'Добавлено вручную');
  renderReview();
  saveSession(currentSession);
});
view.addEventListener('change', () => {
  presets[currentView].direction = directionSelect.value;
  currentView = view.value;
  directionSelect.value = presets[currentView].direction;
  useDefaultLine();
  if (ready && !fileUrl && counter.total === 0) status.textContent = cameraInstructions();
  redraw();
});
directionSelect.addEventListener('change', () => { useDefaultLine(); redraw(); });
get('camera').addEventListener('click', openCamera);
get('file').addEventListener('click', () => get('input').click());
get('input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  const id = resetSource();
  fileUrl = URL.createObjectURL(file);
  videoFilename = file.name;
  video.src = fileUrl;
  try {
    await video.play();
    if (source === id) {
      video.pause();
      video.currentTime = 0;
      prepare(id);
    }
  } catch {
    if (id === source) status.textContent = 'Не удалось открыть видео. Попробуй видео, снятое камерой iPhone.';
  }
});
debug.addEventListener('change', redraw);
video.addEventListener('resize', redraw);
new ResizeObserver(redraw).observe(get('stage'));
openCamera();
