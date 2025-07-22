// Utility functions for data formatting and manipulation

/**
 * Format execution time from milliseconds to human-readable string
 */
export const formatExecutionTime = (ms: number): string => {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(0);
    return `${minutes}m ${seconds}s`;
  }
};

/**
 * Copy data to clipboard
 */
export const copyToClipboard = async (data: any): Promise<boolean> => {
  try {
    const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};

/**
 * Export data as JSON file
 */
export const exportAsJson = (data: any, filename: string): void => {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Format bytes to human-readable string
 */
export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Truncate long strings with ellipsis
 */
export const truncateString = (str: string, maxLength: number): string => {
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength - 3) + '...';
};

/**
 * Extract nested value from object using path
 */
export const getNestedValue = (obj: any, path: string[]): any => {
  return path.reduce((current, key) => current?.[key], obj);
};

/**
 * Flatten nested object for easier searching
 */
export const flattenObject = (
  obj: any,
  prefix = '',
  result: Record<string, any> = {}
): Record<string, any> => {
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const newKey = prefix ? `${prefix}.${key}` : key;
      if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
        flattenObject(obj[key], newKey, result);
      } else {
        result[newKey] = obj[key];
      }
    }
  }
  return result;
};

/**
 * Format timestamp to local datetime string
 */
export const formatTimestamp = (timestamp: number | string): string => {
  const date = new Date(timestamp);
  return date.toLocaleString();
};

/**
 * Calculate diff statistics between two objects
 */
export const calculateDiffStats = (before: any, after: any): {
  added: number;
  removed: number;
  modified: number;
  unchanged: number;
} => {
  const beforeFlat = flattenObject(before);
  const afterFlat = flattenObject(after);
  
  const allKeys = new Set([...Object.keys(beforeFlat), ...Object.keys(afterFlat)]);
  
  let added = 0;
  let removed = 0;
  let modified = 0;
  let unchanged = 0;
  
  allKeys.forEach(key => {
    if (!(key in beforeFlat)) {
      added++;
    } else if (!(key in afterFlat)) {
      removed++;
    } else if (beforeFlat[key] !== afterFlat[key]) {
      modified++;
    } else {
      unchanged++;
    }
  });
  
  return { added, removed, modified, unchanged };
};

/**
 * Generate unique color for a string (for consistent coloring)
 */
export const stringToColor = (str: string): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const h = hash % 360;
  return `hsl(${h}, 70%, 50%)`;
};

/**
 * Parse and format error messages
 */
export const formatError = (error: any): string => {
  if (typeof error === 'string') return error;
  if (error.message) return error.message;
  if (error.error) return formatError(error.error);
  return JSON.stringify(error);
};

/**
 * Debounce function for search and filter operations
 */
export const debounce = <T extends (...args: any[]) => void>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * Format neural data for visualization
 */
export const formatNeuralData = (data: any): {
  type: 'spike_train' | 'heatmap' | 'timeseries';
  formatted: any;
} => {
  if (data.neural_activity) {
    return {
      type: 'heatmap',
      formatted: data.neural_activity.map((activity: any) => ({
        ...activity,
        timestamp: formatTimestamp(activity.timestamp),
      })),
    };
  }
  
  if (data.sdr_data) {
    return {
      type: 'spike_train',
      formatted: data.sdr_data,
    };
  }
  
  return {
    type: 'timeseries',
    formatted: data,
  };
};

/**
 * Convert graph data to standard format for visualization
 */
export const normalizeGraphData = (data: any): {
  nodes: any[];
  edges: any[];
} => {
  // Handle different graph data formats
  if (data.nodes && data.edges) {
    return data;
  }
  
  if (data.vertices && data.links) {
    return {
      nodes: data.vertices,
      edges: data.links,
    };
  }
  
  if (data.entities && data.relationships) {
    return {
      nodes: data.entities.map((e: any) => ({
        id: e.id || e.name,
        type: e.type || 'entity',
        label: e.label || e.name,
        ...e,
      })),
      edges: data.relationships.map((r: any) => ({
        source: r.from || r.source,
        target: r.to || r.target,
        type: r.type || 'relationship',
        ...r,
      })),
    };
  }
  
  // Fallback
  return { nodes: [], edges: [] };
};

/**
 * Sanitize data for display (remove sensitive information)
 */
export const sanitizeData = (data: any, sensitiveKeys: string[] = []): any => {
  const defaultSensitiveKeys = ['password', 'token', 'secret', 'key', 'auth'];
  const allSensitiveKeys = [...defaultSensitiveKeys, ...sensitiveKeys];
  
  const sanitize = (obj: any): any => {
    if (typeof obj !== 'object' || obj === null) return obj;
    
    if (Array.isArray(obj)) {
      return obj.map(sanitize);
    }
    
    const result: any = {};
    for (const key in obj) {
      const lowerKey = key.toLowerCase();
      if (allSensitiveKeys.some(sensitive => lowerKey.includes(sensitive))) {
        result[key] = '[REDACTED]';
      } else {
        result[key] = sanitize(obj[key]);
      }
    }
    return result;
  };
  
  return sanitize(data);
};