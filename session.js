let fallbackSequence = 0;

function sessionId() {
  if (globalThis.crypto?.randomUUID) return `session-${globalThis.crypto.randomUUID()}`;
  const random = new Uint32Array(4);
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(random);
  } else {
    for (let index = 0; index < random.length; index++) random[index] = Math.floor(Math.random() * 0x100000000);
  }
  const entropy = Array.from(random, value => value.toString(16).padStart(8, '0')).join('');
  // Random entropy separates tabs; the sequence also prevents same-tab
  // collisions even if the fallback random source repeats a value.
  return `session-${entropy}-${++fallbackSequence}`;
}

export function createSession(settings, now = Date.now(), id = sessionId()) {
  return {id, startedAt: now, endedAt: null, settings, events: []};
}

export function addAutomaticEvent(session, event, evidence = []) {
  const automaticallyAccepted = event.kind === 'count' && event.reason === 'confirmed-crossing';
  session.events.push({
    ...event, source: 'automatic',
    review: automaticallyAccepted ? 'accepted' : 'pending', evidence,
  });
}

export function addManualEvent(session, time, note) {
  const id = `manual-${session.events.length + 1}-${time}`;
  session.events.push({
    id, kind: 'manual', time, reason: note, source: 'manual',
    review: 'accepted', evidence: [],
  });
  return id;
}

export function reviewEvent(session, id, status) {
  const event = session.events.find(item => item.id === id);
  if (!event) throw new Error(`Unknown event: ${id}`);
  if (!['accepted', 'confirmed', 'removed'].includes(status)) {
    throw new Error(`Invalid review: ${status}`);
  }
  event.review = status;
}

export function totals(session) {
  const automatic = session.events.filter(event => event.source === 'automatic' && event.kind === 'count').length;
  const confirmed = session.events.filter(event => event.review === 'accepted' || event.review === 'confirmed').length;
  return {automatic, confirmed};
}
