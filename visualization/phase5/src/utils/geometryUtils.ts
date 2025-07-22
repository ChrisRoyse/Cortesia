/**
 * Geometry utility functions for Phase 5 visualization
 */

import { Position } from '../types';

/**
 * Calculates the distance between two points
 * @param p1 - First point
 * @param p2 - Second point
 * @returns Distance
 */
export function distance(p1: Position, p2: Position): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculates the angle between two points
 * @param from - Starting point
 * @param to - Ending point
 * @returns Angle in radians
 */
export function angle(from: Position, to: Position): number {
  return Math.atan2(to.y - from.y, to.x - from.x);
}

/**
 * Calculates the midpoint between two points
 * @param p1 - First point
 * @param p2 - Second point
 * @returns Midpoint
 */
export function midpoint(p1: Position, p2: Position): Position {
  return {
    x: (p1.x + p2.x) / 2,
    y: (p1.y + p2.y) / 2
  };
}

/**
 * Calculates a point along a line at a given distance from the start
 * @param start - Starting point
 * @param end - Ending point
 * @param distance - Distance from start
 * @returns Point along the line
 */
export function pointAlongLine(start: Position, end: Position, distance: number): Position {
  const totalDistance = distance(start, end);
  if (totalDistance === 0) return start;
  
  const ratio = distance / totalDistance;
  return {
    x: start.x + (end.x - start.x) * ratio,
    y: start.y + (end.y - start.y) * ratio
  };
}

/**
 * Calculates control points for a quadratic bezier curve
 * @param start - Starting point
 * @param end - Ending point
 * @param curvature - Curvature factor (0 = straight line, 1 = maximum curve)
 * @returns Control point
 */
export function quadraticBezierControl(
  start: Position,
  end: Position,
  curvature: number = 0.5
): Position {
  const mid = midpoint(start, end);
  const dist = distance(start, end);
  const perpAngle = angle(start, end) + Math.PI / 2;
  
  return {
    x: mid.x + Math.cos(perpAngle) * dist * curvature * 0.25,
    y: mid.y + Math.sin(perpAngle) * dist * curvature * 0.25
  };
}

/**
 * Calculates control points for a cubic bezier curve
 * @param start - Starting point
 * @param end - Ending point
 * @param curvature - Curvature factor
 * @returns Two control points
 */
export function cubicBezierControls(
  start: Position,
  end: Position,
  curvature: number = 0.5
): [Position, Position] {
  const dist = distance(start, end);
  const a = angle(start, end);
  
  const cp1: Position = {
    x: start.x + Math.cos(a) * dist * 0.25,
    y: start.y + Math.sin(a) * dist * 0.25 + curvature * dist * 0.25
  };
  
  const cp2: Position = {
    x: end.x - Math.cos(a) * dist * 0.25,
    y: end.y - Math.sin(a) * dist * 0.25 + curvature * dist * 0.25
  };
  
  return [cp1, cp2];
}

/**
 * Checks if a point is inside a circle
 * @param point - Point to check
 * @param center - Center of the circle
 * @param radius - Radius of the circle
 * @returns True if point is inside
 */
export function isPointInCircle(point: Position, center: Position, radius: number): boolean {
  return distance(point, center) <= radius;
}

/**
 * Checks if a point is inside a rectangle
 * @param point - Point to check
 * @param rect - Rectangle bounds
 * @returns True if point is inside
 */
export function isPointInRect(
  point: Position,
  rect: { x: number; y: number; width: number; height: number }
): boolean {
  return (
    point.x >= rect.x &&
    point.x <= rect.x + rect.width &&
    point.y >= rect.y &&
    point.y <= rect.y + rect.height
  );
}

/**
 * Calculates the intersection point of two lines
 * @param line1Start - Start of first line
 * @param line1End - End of first line
 * @param line2Start - Start of second line
 * @param line2End - End of second line
 * @returns Intersection point or null if lines don't intersect
 */
export function lineIntersection(
  line1Start: Position,
  line1End: Position,
  line2Start: Position,
  line2End: Position
): Position | null {
  const x1 = line1Start.x;
  const y1 = line1Start.y;
  const x2 = line1End.x;
  const y2 = line1End.y;
  const x3 = line2Start.x;
  const y3 = line2Start.y;
  const x4 = line2End.x;
  const y4 = line2End.y;
  
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 0.0001) return null; // Lines are parallel
  
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
  
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    return {
      x: x1 + t * (x2 - x1),
      y: y1 + t * (y2 - y1)
    };
  }
  
  return null;
}

/**
 * Rotates a point around a center
 * @param point - Point to rotate
 * @param center - Center of rotation
 * @param angleRad - Angle in radians
 * @returns Rotated point
 */
export function rotatePoint(point: Position, center: Position, angleRad: number): Position {
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  
  return {
    x: center.x + dx * cos - dy * sin,
    y: center.y + dx * sin + dy * cos
  };
}

/**
 * Calculates the bounding box of a set of points
 * @param points - Array of points
 * @returns Bounding box
 */
export function boundingBox(points: Position[]): {
  min: Position;
  max: Position;
  width: number;
  height: number;
  center: Position;
} {
  if (points.length === 0) {
    return {
      min: { x: 0, y: 0 },
      max: { x: 0, y: 0 },
      width: 0,
      height: 0,
      center: { x: 0, y: 0 }
    };
  }
  
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  
  points.forEach(point => {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });
  
  return {
    min: { x: minX, y: minY },
    max: { x: maxX, y: maxY },
    width: maxX - minX,
    height: maxY - minY,
    center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 }
  };
}

/**
 * Generates points along a circle
 * @param center - Center of the circle
 * @param radius - Radius
 * @param count - Number of points
 * @param startAngle - Starting angle in radians
 * @returns Array of points
 */
export function circlePoints(
  center: Position,
  radius: number,
  count: number,
  startAngle: number = 0
): Position[] {
  const points: Position[] = [];
  const angleStep = (2 * Math.PI) / count;
  
  for (let i = 0; i < count; i++) {
    const angle = startAngle + i * angleStep;
    points.push({
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle)
    });
  }
  
  return points;
}

/**
 * Calculates the area of a polygon
 * @param points - Vertices of the polygon
 * @returns Area
 */
export function polygonArea(points: Position[]): number {
  let area = 0;
  const n = points.length;
  
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += points[i].x * points[j].y;
    area -= points[j].x * points[i].y;
  }
  
  return Math.abs(area) / 2;
}

/**
 * Simplifies a path using Douglas-Peucker algorithm
 * @param points - Path points
 * @param epsilon - Maximum distance threshold
 * @returns Simplified path
 */
export function simplifyPath(points: Position[], epsilon: number = 1): Position[] {
  if (points.length <= 2) return points;
  
  // Find the point with maximum distance from the line
  let maxDist = 0;
  let maxIndex = 0;
  
  for (let i = 1; i < points.length - 1; i++) {
    const dist = perpendicularDistance(points[i], points[0], points[points.length - 1]);
    if (dist > maxDist) {
      maxDist = dist;
      maxIndex = i;
    }
  }
  
  // If max distance is greater than epsilon, recursively simplify
  if (maxDist > epsilon) {
    const left = simplifyPath(points.slice(0, maxIndex + 1), epsilon);
    const right = simplifyPath(points.slice(maxIndex), epsilon);
    
    return [...left.slice(0, -1), ...right];
  } else {
    return [points[0], points[points.length - 1]];
  }
}

/**
 * Calculates perpendicular distance from a point to a line
 * @param point - Point to measure from
 * @param lineStart - Start of the line
 * @param lineEnd - End of the line
 * @returns Perpendicular distance
 */
function perpendicularDistance(point: Position, lineStart: Position, lineEnd: Position): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const mag = Math.sqrt(dx * dx + dy * dy);
  
  if (mag === 0) return distance(point, lineStart);
  
  const u = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (mag * mag);
  
  if (u < 0) return distance(point, lineStart);
  if (u > 1) return distance(point, lineEnd);
  
  const intersection = {
    x: lineStart.x + u * dx,
    y: lineStart.y + u * dy
  };
  
  return distance(point, intersection);
}