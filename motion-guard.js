function ignored(x, y, width, height, boxes) {
  const nx = x / width;
  const ny = y / height;
  return boxes.some(([bx, by, bw, bh]) => (
    nx >= bx && nx <= bx + bw && ny >= by && ny <= by + bh
  ));
}

export function estimateTranslation(
  previous,
  current,
  width,
  height,
  {maxShift = 4, ignoredBoxes = []} = {},
) {
  if (previous.length !== current.length || current.length !== width * height) {
    throw new Error('Invalid grayscale frame dimensions');
  }

  let best = {dx: 0, dy: 0, error: Infinity};
  for (let dy = -maxShift; dy <= maxShift; dy += 1) {
    for (let dx = -maxShift; dx <= maxShift; dx += 1) {
      let difference = 0;
      let samples = 0;
      for (let y = maxShift; y < height - maxShift; y += 2) {
        for (let x = maxShift; x < width - maxShift; x += 2) {
          if (ignored(x, y, width, height, ignoredBoxes)) continue;

          const px = x - dx;
          const py = y - dy;
          if (px < 0 || px >= width || py < 0 || py >= height) continue;

          difference += Math.abs(current[y * width + x] - previous[py * width + px]);
          samples += 1;
        }
      }

      const error = samples ? difference / samples : Infinity;
      const displacement = dx * dx + dy * dy;
      const bestDisplacement = best.dx * best.dx + best.dy * best.dy;
      const tiedWithSmallerDisplacement = Number.isFinite(error)
        && error === best.error
        && displacement < bestDisplacement;
      const tiedWithSameDisplacement = Number.isFinite(error)
        && error === best.error
        && displacement === bestDisplacement
        && (dy < best.dy || (dy === best.dy && dx < best.dx));
      if (error < best.error || tiedWithSmallerDisplacement || tiedWithSameDisplacement) {
        best = {dx, dy, error};
      }
    }
  }
  return best;
}

export class MotionGuard {
  constructor({blockPixels = 2, stableFrames = 3, maxShift = 4} = {}) {
    this.blockPixels = blockPixels;
    this.requiredStableFrames = stableFrames;
    this.maxShift = maxShift;
    this.reset();
  }

  reset() {
    this.blocked = false;
    this.stable = 0;
  }

  update(previous, current, width, height, ignoredBoxes = []) {
    const result = estimateTranslation(previous, current, width, height, {
      maxShift: this.maxShift,
      ignoredBoxes,
    });
    if (result.error === Infinity) return {...result, blocked: this.blocked};

    const moving = Math.hypot(result.dx, result.dy) >= this.blockPixels;

    if (moving) {
      this.blocked = true;
      this.stable = 0;
    } else if (this.blocked) {
      this.stable += 1;
      if (this.stable >= this.requiredStableFrames) this.blocked = false;
    }

    return {...result, blocked: this.blocked};
  }
}
