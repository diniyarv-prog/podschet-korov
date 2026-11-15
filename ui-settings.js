const KEY = 'cow-counter-ui-v1';
const views = new Set(['side', 'overhead']);
const directions = new Set(['right', 'left', 'down', 'up']);

export const DEFAULT_UI_SETTINGS = Object.freeze({
  view: 'side',
  direction: 'right',
  line: {start: [0.5, 0.15], end: [0.5, 0.85]},
});

const point = value => Array.isArray(value) && value.length === 2 &&
  value.every(number => Number.isFinite(number) && number >= 0 && number <= 1);

function valid(value) {
  if (!value || !views.has(value.view) || !directions.has(value.direction) ||
      !point(value.line?.start) || !point(value.line?.end)) return false;
  return Math.hypot(
    value.line.start[0] - value.line.end[0],
    value.line.start[1] - value.line.end[1],
  ) >= 0.01;
}

const clone = value => structuredClone(value);

export function loadUiSettings(storage) {
  try {
    const value = JSON.parse(storage?.getItem(KEY) ?? 'null');
    return valid(value) ? clone(value) : clone(DEFAULT_UI_SETTINGS);
  } catch {
    return clone(DEFAULT_UI_SETTINGS);
  }
}

export function saveUiSettings(storage, settings) {
  const value = valid(settings) ? clone(settings) : clone(DEFAULT_UI_SETTINGS);
  try {
    storage?.setItem(KEY, JSON.stringify(value));
  } catch {
    // Storage can be unavailable in private browsing; the live setup remains usable.
  }
  return value;
}
