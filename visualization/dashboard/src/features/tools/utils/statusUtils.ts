import { ToolStatus, ToolStatusInfo, MCPTool, ToolCategory } from '../types';
import { StatusHistory } from '../services/ToolStatusMonitor';

// Status color mappings
export const STATUS_COLORS = {
  healthy: '#4caf50',
  degraded: '#ff9800',
  unavailable: '#f44336',
  unknown: '#9e9e9e'
} as const;

// Status priority for sorting
export const STATUS_PRIORITY: Record<ToolStatus, number> = {
  unavailable: 0,
  degraded: 1,
  unknown: 2,
  healthy: 3
};

// Response time thresholds (in milliseconds)
export const RESPONSE_TIME_THRESHOLDS = {
  excellent: 50,
  good: 200,
  acceptable: 500,
  poor: 1000
} as const;

// Error rate thresholds (as percentages)
export const ERROR_RATE_THRESHOLDS = {
  excellent: 0.1,  // 0.1%
  good: 1,         // 1%
  acceptable: 5,   // 5%
  poor: 10         // 10%
} as const;

// Calculate health score from metrics
export function calculateHealthScore(status: ToolStatusInfo): number {
  let score = 0;
  
  // Response time scoring (0-40 points)
  if (status.responseTime < RESPONSE_TIME_THRESHOLDS.excellent) {
    score += 40;
  } else if (status.responseTime < RESPONSE_TIME_THRESHOLDS.good) {
    score += 30;
  } else if (status.responseTime < RESPONSE_TIME_THRESHOLDS.acceptable) {
    score += 20;
  } else if (status.responseTime < RESPONSE_TIME_THRESHOLDS.poor) {
    score += 10;
  }
  
  // Error rate scoring (0-40 points)
  const errorRatePercent = status.errorRate * 100;
  if (errorRatePercent < ERROR_RATE_THRESHOLDS.excellent) {
    score += 40;
  } else if (errorRatePercent < ERROR_RATE_THRESHOLDS.good) {
    score += 30;
  } else if (errorRatePercent < ERROR_RATE_THRESHOLDS.acceptable) {
    score += 20;
  } else if (errorRatePercent < ERROR_RATE_THRESHOLDS.poor) {
    score += 10;
  }
  
  // Availability scoring (0-20 points)
  if (status.available) {
    score += 20;
  }
  
  return score;
}

// Determine status from health score
export function getStatusFromScore(score: number): ToolStatus {
  if (score >= 80) return 'healthy';
  if (score >= 50) return 'degraded';
  if (score > 0) return 'unavailable';
  return 'unknown';
}

// Format response time for display
export function formatResponseTime(ms: number): string {
  if (ms < 1) return '<1ms';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// Format error rate for display
export function formatErrorRate(rate: number): string {
  const percentage = rate * 100;
  if (percentage === 0) return '0%';
  if (percentage < 0.1) return '<0.1%';
  return `${percentage.toFixed(1)}%`;
}

// Format availability for display
export function formatAvailability(available: boolean, uptimePercent?: number): string {
  if (uptimePercent !== undefined) {
    return `${uptimePercent.toFixed(2)}%`;
  }
  return available ? 'Available' : 'Unavailable';
}

// Calculate uptime from history
export function calculateUptime(history: StatusHistory[], periodHours: number = 24): number {
  if (history.length === 0) return 0;
  
  const now = Date.now();
  const periodMs = periodHours * 60 * 60 * 1000;
  const cutoff = now - periodMs;
  
  // Filter history to period
  const relevantHistory = history.filter(h => h.timestamp.getTime() >= cutoff);
  if (relevantHistory.length === 0) return 0;
  
  // Calculate total available time
  let availableTime = 0;
  let lastTimestamp = cutoff;
  
  relevantHistory.forEach(entry => {
    const entryTime = entry.timestamp.getTime();
    if (entry.available) {
      availableTime += entryTime - lastTimestamp;
    }
    lastTimestamp = entryTime;
  });
  
  // Add time from last entry to now if available
  const lastEntry = relevantHistory[relevantHistory.length - 1];
  if (lastEntry.available) {
    availableTime += now - lastEntry.timestamp.getTime();
  }
  
  return (availableTime / periodMs) * 100;
}

// Calculate average response time from history
export function calculateAverageResponseTime(history: StatusHistory[]): number {
  if (history.length === 0) return 0;
  
  const total = history.reduce((sum, entry) => sum + entry.responseTime, 0);
  return total / history.length;
}

// Calculate percentile response time
export function calculatePercentileResponseTime(history: StatusHistory[], percentile: number): number {
  if (history.length === 0) return 0;
  
  const sorted = history.map(h => h.responseTime).sort((a, b) => a - b);
  const index = Math.ceil((percentile / 100) * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

// Get status trend from history
export function getStatusTrend(history: StatusHistory[]): 'improving' | 'stable' | 'degrading' | 'unknown' {
  if (history.length < 2) return 'unknown';
  
  // Compare recent history to older history
  const midpoint = Math.floor(history.length / 2);
  const oldHistory = history.slice(0, midpoint);
  const recentHistory = history.slice(midpoint);
  
  const oldScore = oldHistory.reduce((sum, h) => sum + scoreStatus(h.status), 0) / oldHistory.length;
  const recentScore = recentHistory.reduce((sum, h) => sum + scoreStatus(h.status), 0) / recentHistory.length;
  
  const difference = recentScore - oldScore;
  
  if (difference > 0.5) return 'improving';
  if (difference < -0.5) return 'degrading';
  return 'stable';
}

// Score status for trend calculation
function scoreStatus(status: ToolStatus): number {
  const scores: Record<ToolStatus, number> = {
    healthy: 3,
    degraded: 2,
    unavailable: 1,
    unknown: 0
  };
  return scores[status];
}

// Group tools by status
export function groupToolsByStatus(tools: MCPTool[]): Record<ToolStatus, MCPTool[]> {
  const groups: Record<ToolStatus, MCPTool[]> = {
    healthy: [],
    degraded: [],
    unavailable: [],
    unknown: []
  };
  
  tools.forEach(tool => {
    groups[tool.status.health].push(tool);
  });
  
  return groups;
}

// Get category health summary
export function getCategoryHealthSummary(tools: MCPTool[]): Record<ToolCategory, {
  total: number;
  healthy: number;
  degraded: number;
  unavailable: number;
  healthPercent: number;
}> {
  const summary: Record<string, any> = {};
  
  tools.forEach(tool => {
    if (!summary[tool.category]) {
      summary[tool.category] = {
        total: 0,
        healthy: 0,
        degraded: 0,
        unavailable: 0,
        healthPercent: 0
      };
    }
    
    const cat = summary[tool.category];
    cat.total++;
    
    switch (tool.status.health) {
      case 'healthy':
        cat.healthy++;
        break;
      case 'degraded':
        cat.degraded++;
        break;
      case 'unavailable':
        cat.unavailable++;
        break;
    }
  });
  
  // Calculate health percentages
  Object.values(summary).forEach(cat => {
    cat.healthPercent = cat.total > 0 ? (cat.healthy / cat.total) * 100 : 0;
  });
  
  return summary;
}

// Generate status report
export function generateStatusReport(tools: MCPTool[], histories: Map<string, StatusHistory[]>): {
  summary: {
    totalTools: number;
    healthyTools: number;
    degradedTools: number;
    unavailableTools: number;
    overallHealth: number;
    averageResponseTime: number;
    averageErrorRate: number;
  };
  categoryBreakdown: ReturnType<typeof getCategoryHealthSummary>;
  criticalTools: MCPTool[];
  trends: Map<string, ReturnType<typeof getStatusTrend>>;
} {
  const statusGroups = groupToolsByStatus(tools);
  const categoryBreakdown = getCategoryHealthSummary(tools);
  
  // Calculate averages
  let totalResponseTime = 0;
  let totalErrorRate = 0;
  let count = 0;
  
  tools.forEach(tool => {
    if (tool.status.responseTime) {
      totalResponseTime += tool.status.responseTime;
      totalErrorRate += tool.status.errorRate;
      count++;
    }
  });
  
  // Identify critical tools (unavailable or high error rate)
  const criticalTools = tools.filter(tool => 
    tool.status.health === 'unavailable' || 
    tool.status.errorRate > ERROR_RATE_THRESHOLDS.acceptable / 100
  );
  
  // Calculate trends
  const trends = new Map<string, ReturnType<typeof getStatusTrend>>();
  tools.forEach(tool => {
    const history = histories.get(tool.id) || [];
    trends.set(tool.id, getStatusTrend(history));
  });
  
  return {
    summary: {
      totalTools: tools.length,
      healthyTools: statusGroups.healthy.length,
      degradedTools: statusGroups.degraded.length,
      unavailableTools: statusGroups.unavailable.length,
      overallHealth: tools.length > 0 ? (statusGroups.healthy.length / tools.length) * 100 : 0,
      averageResponseTime: count > 0 ? totalResponseTime / count : 0,
      averageErrorRate: count > 0 ? totalErrorRate / count : 0
    },
    categoryBreakdown,
    criticalTools,
    trends
  };
}

// Export status data to CSV
export function exportStatusToCSV(tools: MCPTool[]): string {
  const headers = [
    'Tool ID',
    'Tool Name',
    'Category',
    'Status',
    'Response Time (ms)',
    'Error Rate (%)',
    'Available',
    'Last Checked',
    'Total Executions',
    'Success Rate (%)'
  ];
  
  const rows = tools.map(tool => [
    tool.id,
    tool.name,
    tool.category,
    tool.status.health,
    tool.status.responseTime.toFixed(2),
    (tool.status.errorRate * 100).toFixed(2),
    tool.status.available ? 'Yes' : 'No',
    tool.status.lastChecked.toISOString(),
    tool.metrics.totalExecutions,
    tool.metrics.successRate.toFixed(2)
  ]);
  
  const csv = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');
  
  return csv;
}

// Check if tool needs attention
export function toolNeedsAttention(tool: MCPTool): boolean {
  return (
    tool.status.health === 'unavailable' ||
    tool.status.health === 'degraded' ||
    tool.status.errorRate > ERROR_RATE_THRESHOLDS.acceptable / 100 ||
    tool.status.responseTime > RESPONSE_TIME_THRESHOLDS.acceptable
  );
}

// Get health emoji
export function getHealthEmoji(status: ToolStatus): string {
  const emojis: Record<ToolStatus, string> = {
    healthy: '✅',
    degraded: '⚠️',
    unavailable: '❌',
    unknown: '❓'
  };
  return emojis[status];
}

// Format status message
export function formatStatusMessage(tool: MCPTool): string {
  const emoji = getHealthEmoji(tool.status.health);
  const responseTime = formatResponseTime(tool.status.responseTime);
  const errorRate = formatErrorRate(tool.status.errorRate);
  
  return `${emoji} ${tool.name} - ${tool.status.health} (${responseTime}, ${errorRate} errors)`;
}