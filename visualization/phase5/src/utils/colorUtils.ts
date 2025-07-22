/**
 * Color utility functions for Phase 5 visualization
 */

/**
 * Mixes two colors together with a given ratio
 * @param color1 - First color in hex format
 * @param color2 - Second color in hex format
 * @param ratio - Mix ratio (0 = all color1, 1 = all color2)
 * @returns Mixed color in hex format
 */
export function mixColors(color1: string, color2: string, ratio: number): string {
  // Convert hex to RGB
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);
  
  if (!rgb1 || !rgb2) {
    return color1; // Fallback to first color if parsing fails
  }
  
  // Mix the colors
  const r = Math.round(rgb1.r * (1 - ratio) + rgb2.r * ratio);
  const g = Math.round(rgb1.g * (1 - ratio) + rgb2.g * ratio);
  const b = Math.round(rgb1.b * (1 - ratio) + rgb2.b * ratio);
  
  return rgbToHex(r, g, b);
}

/**
 * Desaturates a color by a given amount
 * @param color - Color in hex format
 * @param amount - Desaturation amount (0 = no change, 1 = fully grayscale)
 * @returns Desaturated color in hex format
 */
export function desaturateColor(color: string, amount: number): string {
  const rgb = hexToRgb(color);
  if (!rgb) return color;
  
  // Calculate grayscale value
  const gray = rgb.r * 0.299 + rgb.g * 0.587 + rgb.b * 0.114;
  
  // Mix with grayscale
  const r = Math.round(rgb.r * (1 - amount) + gray * amount);
  const g = Math.round(rgb.g * (1 - amount) + gray * amount);
  const b = Math.round(rgb.b * (1 - amount) + gray * amount);
  
  return rgbToHex(r, g, b);
}

/**
 * Adjusts the brightness of a color
 * @param color - Color in hex format
 * @param amount - Brightness adjustment (-1 to 1, negative = darker, positive = lighter)
 * @returns Adjusted color in hex format
 */
export function adjustBrightness(color: string, amount: number): string {
  const rgb = hexToRgb(color);
  if (!rgb) return color;
  
  const adjust = amount * 255;
  
  const r = Math.max(0, Math.min(255, rgb.r + adjust));
  const g = Math.max(0, Math.min(255, rgb.g + adjust));
  const b = Math.max(0, Math.min(255, rgb.b + adjust));
  
  return rgbToHex(Math.round(r), Math.round(g), Math.round(b));
}

/**
 * Converts a hex color to RGB
 * @param hex - Hex color string
 * @returns RGB object or null if invalid
 */
export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  // Remove # if present
  hex = hex.replace(/^#/, '');
  
  // Handle 3-digit hex
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  
  if (hex.length !== 6) {
    return null;
  }
  
  const num = parseInt(hex, 16);
  
  if (isNaN(num)) {
    return null;
  }
  
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255
  };
}

/**
 * Converts RGB values to hex color
 * @param r - Red value (0-255)
 * @param g - Green value (0-255)
 * @param b - Blue value (0-255)
 * @returns Hex color string
 */
export function rgbToHex(r: number, g: number, b: number): string {
  const toHex = (n: number) => {
    const hex = Math.max(0, Math.min(255, Math.round(n))).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  
  return '#' + toHex(r) + toHex(g) + toHex(b);
}

/**
 * Gets a color with adjusted opacity
 * @param color - Color in hex format
 * @param opacity - Opacity value (0-1)
 * @returns RGBA color string
 */
export function withOpacity(color: string, opacity: number): string {
  const rgb = hexToRgb(color);
  if (!rgb) return color;
  
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
}

/**
 * Generates a color scale between two colors
 * @param startColor - Starting color in hex format
 * @param endColor - Ending color in hex format
 * @param steps - Number of steps in the scale
 * @returns Array of colors in hex format
 */
export function generateColorScale(startColor: string, endColor: string, steps: number): string[] {
  const scale: string[] = [];
  
  for (let i = 0; i < steps; i++) {
    const ratio = i / (steps - 1);
    scale.push(mixColors(startColor, endColor, ratio));
  }
  
  return scale;
}

/**
 * Gets a contrasting text color (black or white) for a given background
 * @param backgroundColor - Background color in hex format
 * @returns '#000000' or '#ffffff' for optimal contrast
 */
export function getContrastColor(backgroundColor: string): string {
  const rgb = hexToRgb(backgroundColor);
  if (!rgb) return '#000000';
  
  // Calculate relative luminance
  const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
  
  return luminance > 0.5 ? '#000000' : '#ffffff';
}