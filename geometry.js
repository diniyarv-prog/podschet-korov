export const directionVectors = {
  right: [1, 0], left: [-1, 0], down: [0, 1], up: [0, -1]
};

export function boxCenter([x, y, width, height]) {
  return [x + width / 2, y + height / 2];
}

export function orientedDistance([px, py], {start, end}, direction) {
  const [ax, ay] = start;
  const [bx, by] = end;
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.hypot(dx, dy);
  if (length < 0.001) throw new Error('Counting line is too short');
  let nx = -dy / length;
  let ny = dx / length;
  const [tx, ty] = directionVectors[direction];
  if (nx * tx + ny * ty < 0) {
    nx *= -1;
    ny *= -1;
  }
  return (px - ax) * nx + (py - ay) * ny;
}
