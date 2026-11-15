export class EvidenceBuffer {
  constructor(limit = 6) {
    this.limit = limit;
    this.reset();
  }

  reset() {
    this.frames = [];
    this.pending = new Map();
  }

  push(frame) {
    this.frames.push(frame);
    if (this.frames.length > this.limit) this.frames.shift();
  }

  capture(eventId, currentFrame) {
    const evidence = {
      before: this.frames[0] ?? currentFrame,
      during: currentFrame,
      after: null,
    };
    this.pending.set(eventId, evidence);
    return evidence;
  }

  complete(eventId, afterFrame) {
    const evidence = this.pending.get(eventId);
    if (!evidence) return null;
    evidence.after = afterFrame;
    this.pending.delete(eventId);
    return evidence;
  }
}
