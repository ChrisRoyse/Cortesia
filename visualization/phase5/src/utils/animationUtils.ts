/**
 * Animation utility functions for Phase 5 visualization
 */

/**
 * Easing functions for animations
 */
export const easings = {
  linear: (t: number) => t,
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number) => t * t * t,
  easeOutCubic: (t: number) => (--t) * t * t + 1,
  easeInOutCubic: (t: number) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  easeInElastic: (t: number) => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
  },
  easeOutElastic: (t: number) => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
  easeOutBounce: (t: number) => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      return n1 * (t -= 1.5 / d1) * t + 0.75;
    } else if (t < 2.5 / d1) {
      return n1 * (t -= 2.25 / d1) * t + 0.9375;
    } else {
      return n1 * (t -= 2.625 / d1) * t + 0.984375;
    }
  }
};

/**
 * Creates a staggered delay for animations
 * @param index - Index of the element
 * @param baseDelay - Base delay in milliseconds
 * @param staggerAmount - Amount to stagger in milliseconds
 * @returns Calculated delay
 */
export function staggerDelay(index: number, baseDelay: number = 0, staggerAmount: number = 50): number {
  return baseDelay + (index * staggerAmount);
}

/**
 * Interpolates between two values
 * @param from - Starting value
 * @param to - Ending value
 * @param progress - Progress (0-1)
 * @returns Interpolated value
 */
export function interpolate(from: number, to: number, progress: number): number {
  return from + (to - from) * progress;
}

/**
 * Interpolates between two positions
 * @param from - Starting position
 * @param to - Ending position
 * @param progress - Progress (0-1)
 * @returns Interpolated position
 */
export function interpolatePosition(
  from: { x: number; y: number },
  to: { x: number; y: number },
  progress: number
): { x: number; y: number } {
  return {
    x: interpolate(from.x, to.x, progress),
    y: interpolate(from.y, to.y, progress)
  };
}

/**
 * Creates a spring physics animation configuration
 * @param tension - Spring tension (default: 170)
 * @param friction - Spring friction (default: 26)
 * @returns Spring configuration
 */
export function springConfig(tension: number = 170, friction: number = 26) {
  return {
    tension,
    friction,
    mass: 1,
    velocity: 0
  };
}

/**
 * Calculates the duration for a distance-based animation
 * @param distance - Distance to travel
 * @param speed - Speed in units per second
 * @param minDuration - Minimum duration in milliseconds
 * @param maxDuration - Maximum duration in milliseconds
 * @returns Duration in milliseconds
 */
export function calculateAnimationDuration(
  distance: number,
  speed: number = 1000,
  minDuration: number = 200,
  maxDuration: number = 2000
): number {
  const calculatedDuration = (distance / speed) * 1000;
  return Math.max(minDuration, Math.min(maxDuration, calculatedDuration));
}

/**
 * Creates a pulse animation keyframe
 * @param scale - Maximum scale
 * @param duration - Duration in milliseconds
 * @returns Animation keyframes
 */
export function createPulseAnimation(scale: number = 1.2, duration: number = 1000) {
  return {
    keyframes: [
      { transform: 'scale(1)', opacity: 1 },
      { transform: `scale(${scale})`, opacity: 0.7 },
      { transform: 'scale(1)', opacity: 1 }
    ],
    options: {
      duration,
      iterations: Infinity,
      easing: 'ease-in-out'
    }
  };
}

/**
 * Creates a ripple effect animation
 * @param origin - Origin point of the ripple
 * @param maxRadius - Maximum radius of the ripple
 * @param duration - Duration in milliseconds
 * @returns Animation configuration
 */
export function createRippleAnimation(
  origin: { x: number; y: number },
  maxRadius: number = 100,
  duration: number = 1000
) {
  return {
    keyframes: [
      { 
        transform: `translate(${origin.x}px, ${origin.y}px) scale(0)`,
        opacity: 0.8
      },
      { 
        transform: `translate(${origin.x - maxRadius}px, ${origin.y - maxRadius}px) scale(1)`,
        opacity: 0
      }
    ],
    options: {
      duration,
      easing: 'ease-out'
    }
  };
}

/**
 * Debounces an animation frame request
 * @param callback - Function to debounce
 * @returns Debounced function
 */
export function debounceAnimationFrame<T extends (...args: any[]) => void>(
  callback: T
): (...args: Parameters<T>) => void {
  let animationFrameId: number | null = null;
  
  return (...args: Parameters<T>) => {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
    }
    
    animationFrameId = requestAnimationFrame(() => {
      callback(...args);
      animationFrameId = null;
    });
  };
}

/**
 * Creates a smooth scroll configuration
 * @param target - Target scroll position
 * @param duration - Duration in milliseconds
 * @param easing - Easing function name
 * @returns Scroll configuration
 */
export function smoothScrollConfig(
  target: number,
  duration: number = 500,
  easing: keyof typeof easings = 'easeInOutQuad'
) {
  return {
    target,
    duration,
    easing: easings[easing]
  };
}

/**
 * Animates a numeric value over time
 * @param from - Starting value
 * @param to - Ending value
 * @param duration - Duration in milliseconds
 * @param onUpdate - Callback for each frame
 * @param easing - Easing function
 * @returns Cleanup function
 */
export function animateValue(
  from: number,
  to: number,
  duration: number,
  onUpdate: (value: number) => void,
  easing: (t: number) => number = easings.easeInOutQuad
): () => void {
  const startTime = performance.now();
  let animationFrameId: number;
  
  const animate = (currentTime: number) => {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easedProgress = easing(progress);
    const currentValue = interpolate(from, to, easedProgress);
    
    onUpdate(currentValue);
    
    if (progress < 1) {
      animationFrameId = requestAnimationFrame(animate);
    }
  };
  
  animationFrameId = requestAnimationFrame(animate);
  
  return () => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
  };
}