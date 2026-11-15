import { boxCenter, orientedDistance } from './geometry.js';

export class CrossingCounter {
  constructor({entryDistance = 0.035, exitDistance = 0.035, minHits = 3, stateRetentionMs = 1300} = {}) {
    this.entryDistance = entryDistance;
    this.exitDistance = exitDistance;
    this.minHits = minHits;
    this.stateRetentionMs = stateRetentionMs;
    this.reset();
  }

  reset() {
    this.states = new Map();
    this.total = 0;
    this.sequence = 0;
  }

  update(tracks, time, line, direction, {counting = true, motionBlocked = false} = {}) {
    if (!counting) return [];
    const automaticDirection = direction === 'auto';
    const events = [];
    for (const [id, state] of this.states) {
      if (time - state.lastObservedTime > this.stateRetentionMs) this.states.delete(id);
    }
    const observed = new Map(tracks.filter(track => track.confirmed && track.updated)
      .map(track => [track.id, {track, isObserved: true}]));
    // Camera motion interrupts every pending trajectory, including a track
    // temporarily missing from the detector's current frame.
    if (motionBlocked) {
      for (const [id, state] of this.states) {
        if (!observed.has(id)) observed.set(id, {track: state.lastTrack, isObserved: false});
      }
    }
    for (const {track, isObserved} of observed.values()) {
      const position = orientedDistance(boxCenter(track.box), line, automaticDirection ? 'right' : direction);
      const state = this.states.get(track.id) ?? {
        approached: false, crossed: false, counted: false,
        uncertainSent: false, motionCandidate: false, approachSign: 0,
        lastPosition: position, lastObservedTime: time
      };
      // Both outcomes are terminal: review may change an event, never the
      // number of passages contributed by this track.
      if (state.counted || state.uncertainSent) {
        // Keep a live tracker identity terminal across its normal retention
        // period, but never extend it from an injected historical frame.
        if (isObserved) {
          state.lastPosition = position;
          state.lastTrack = track;
          state.lastObservedTime = time;
          this.states.set(track.id, state);
        }
        continue;
      }
      if (motionBlocked) {
        state.motionCandidate ||= automaticDirection ? state.approachSign !== 0 : state.approached;
        const reachedZone = state.crossed || (automaticDirection
          ? Math.min(Math.abs(state.lastPosition), Math.abs(position)) <= this.entryDistance ||
            state.lastPosition * position <= 0
          : Math.max(state.lastPosition, position) >= -this.entryDistance);
        if (state.motionCandidate && reachedZone) {
          state.uncertainSent = true;
          events.push(this.event(track, time, 'uncertain', 'camera-motion'));
        }
        // A later count needs a fresh approach wholly observed after motion.
        state.approached = false;
        state.approachSign = 0;
        state.crossed = false;
      } else if (automaticDirection) {
        state.motionCandidate = false;
        if (!state.approachSign) {
          if (position <= -this.entryDistance) state.approachSign = -1;
          else if (position >= this.entryDistance) state.approachSign = 1;
        }
        if ((state.approachSign < 0 && state.lastPosition <= 0 && position >= 0) ||
            (state.approachSign > 0 && state.lastPosition >= 0 && position <= 0)) {
          state.crossed = true;
        }
      } else {
        state.motionCandidate = false;
        if (position <= -this.entryDistance) state.approached = true;
        if (state.approached && state.lastPosition <= 0 && position >= 0) state.crossed = true;
      }
      const reachedExit = automaticDirection
        ? (state.approachSign < 0 && position >= this.exitDistance) ||
          (state.approachSign > 0 && position <= -this.exitDistance)
        : position >= this.exitDistance;
      if (!motionBlocked && state.crossed && !state.counted &&
          reachedExit && track.hits >= this.minHits) {
        state.counted = true;
        this.total += 1;
        events.push(this.event(track, time, 'count', this.reason(track)));
      }
      state.lastPosition = position;
      state.lastTrack = track;
      if (isObserved) state.lastObservedTime = time;
      this.states.set(track.id, state);
    }
    return events;
  }

  event(track, time, kind, reason) {
    return {
      id: `event-${++this.sequence}`,
      trackId: track.id,
      time,
      kind,
      reason,
      confidence: Math.max(0, Math.min(1, track.score ?? 0))
    };
  }

  reason(track) {
    if ((track.maxGapMs ?? 0) > 400) return 'track-gap';
    if ((track.score ?? 0) < 0.35) return 'low-confidence';
    return 'confirmed-crossing';
  }
}
