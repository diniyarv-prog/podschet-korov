import { boxCenter } from './geometry.js';

function iou(first, second) {
  const [ax, ay, aw, ah] = first;
  const [bx, by, bw, bh] = second;
  const w = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx));
  const h = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
  const overlap = w * h;
  return overlap / Math.max(aw * ah + bw * bh - overlap, 1e-9);
}

function projectedBox(track, time) {
  const elapsed = Math.min(Math.max(time - track.seen, 0), 550);
  const box = [...track.box];
  box[0] += track.vx * elapsed;
  box[1] += track.vy * elapsed;
  return box;
}

function associate(tracks, detections, time, usedTracks, usedDetections) {
  const candidates = [];
  tracks.forEach(track => {
    if (usedTracks.has(track)) return;
    const projected = projectedBox(track, time);
    const [px, py] = boxCenter(projected);
    detections.forEach((detection, index) => {
      if (usedDetections.has(index)) return;
      const [dx, dy] = boxCenter(detection.bbox);
      const distance = Math.hypot(px - dx, py - dy);
      const overlap = iou(projected, detection.bbox);
      const gate = Math.max(0.08, Math.min(0.24, Math.max(track.box[2], track.box[3]) * 0.9));
      if (distance <= gate || overlap >= 0.1) {
        candidates.push({track, index, cost: distance / gate + (1 - overlap) * 0.4});
      }
    });
  });
  candidates.sort((a, b) => a.cost - b.cost);
  const matches = [];
  for (const candidate of candidates) {
    if (usedTracks.has(candidate.track) || usedDetections.has(candidate.index)) continue;
    usedTracks.add(candidate.track);
    usedDetections.add(candidate.index);
    matches.push(candidate);
  }
  return matches;
}

export class CowTracker {
  constructor({strongScore = 0.45, weakScore = 0.18, maxAgeMs = 1300} = {}) {
    this.strongScore = strongScore;
    this.weakScore = weakScore;
    this.maxAgeMs = maxAgeMs;
    this.reset();
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(detections, time) {
    this.tracks = this.tracks.filter(track => time - track.seen <= this.maxAgeMs);
    this.tracks.forEach(track => { track.updated = false; });
    const strong = detections.filter(item => item.score >= this.strongScore);
    const weak = detections.filter(item => item.score >= this.weakScore && item.score < this.strongScore);
    const usedTracks = new Set();
    const usedStrong = new Set();
    const usedWeak = new Set();
    const apply = (items, matches) => matches.forEach(({track, index}) => this.apply(track, items[index], time));
    apply(strong, associate(this.tracks, strong, time, usedTracks, usedStrong));
    apply(weak, associate(this.tracks, weak, time, usedTracks, usedWeak));
    strong.forEach((item, index) => {
      if (!usedStrong.has(index)) this.create(item, time);
    });
    return this.tracks.filter(track => track.updated && track.confirmed);
  }

  create(detection, time) {
    this.tracks.push({
      id: this.nextId++, box: detection.bbox, score: detection.score,
      seen: time, vx: 0, vy: 0, hits: 1, confirmed: false,
      updated: true, maxGapMs: 0,
      history: [{point: boxCenter(detection.bbox), time}]
    });
  }

  apply(track, detection, time) {
    const [x, y] = boxCenter(detection.bbox);
    const [oldX, oldY] = boxCenter(track.box);
    const elapsed = Math.max(time - track.seen, 1);
    track.maxGapMs = Math.max(track.maxGapMs, elapsed);
    track.vx = 0.4 * track.vx + 0.6 * (x - oldX) / elapsed;
    track.vy = 0.4 * track.vy + 0.6 * (y - oldY) / elapsed;
    track.box = detection.bbox;
    track.score = detection.score;
    track.seen = time;
    track.hits += 1;
    track.confirmed ||= track.hits >= 2;
    track.updated = true;
    track.history.push({point: [x, y], time});
    track.history = track.history.slice(-12);
  }
}
