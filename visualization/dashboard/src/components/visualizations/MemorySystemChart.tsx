import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useAppSelector } from '../../stores';
import { MemoryData, MemoryStore, PerformanceMetrics } from '../../types';

interface MemorySystemChartProps {
  memoryData: MemoryData;
  width?: number;
  height?: number;
  timeRange?: number;
  showTrends?: boolean;
  showBreakdown?: boolean;
  interactive?: boolean;
  className?: string;
}

interface MemoryTimePoint {
  timestamp: number;
  usage: number;
  available: number;
  performance: PerformanceMetrics;
  stores: MemoryStore[];
}

interface ChartSection {
  name: string;
  height: number;
  yOffset: number;
  render: (g: d3.Selection<SVGGElement, unknown, null, undefined>) => void;
}

// Chart configuration
const CHART_CONFIG = {
  margin: { top: 40, right: 80, bottom: 60, left: 80 },
  sections: {
    usage: { height: 0.4, title: 'Memory Usage' },
    performance: { height: 0.3, title: 'Performance Metrics' },
    stores: { height: 0.3, title: 'Memory Stores' },
  },
  colors: {
    usage: '#4ecdc4',
    available: '#a8e6cf',
    performance: {
      latency: '#ff6b6b',
      throughput: '#4ecdc4',
      errorRate: '#ffa726',
      uptime: '#66bb6a',
    },
    stores: {
      sdr: '#61dafb',
      zce: '#f093fb',
      cache: '#a8e6cf',
    },
  },
  animation: {
    duration: 750,
    ease: d3.easeElasticOut,
  },
  thresholds: {
    memory: {
      warning: 0.7,
      critical: 0.9,
    },
    performance: {
      latency: { warning: 100, critical: 500 },
      errorRate: { warning: 0.01, critical: 0.05 },
    },
  },
};

const MemorySystemChart: React.FC<MemorySystemChartProps> = ({
  memoryData,
  width = 800,
  height = 600,
  timeRange = 60000, // 1 minute
  showTrends = true,
  showBreakdown = true,
  interactive = true,
  className = '',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const theme = useAppSelector(state => state.dashboard.config.theme);
  const enableAnimations = useAppSelector(state => state.dashboard.config.enableAnimations);
  
  const [memoryHistory, setMemoryHistory] = useState<MemoryTimePoint[]>([]);
  const [hoveredData, setHoveredData] = useState<any>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  // Calculate chart dimensions
  const chartDimensions = useMemo(() => {
    return {
      width: width - CHART_CONFIG.margin.left - CHART_CONFIG.margin.right,
      height: height - CHART_CONFIG.margin.top - CHART_CONFIG.margin.bottom,
    };
  }, [width, height]);

  // Update memory history
  useEffect(() => {
    if (memoryData) {
      const currentTime = Date.now();
      const newDataPoint: MemoryTimePoint = {
        timestamp: currentTime,
        usage: memoryData.usage.percentage,
        available: 1 - memoryData.usage.percentage,
        performance: memoryData.performance,
        stores: memoryData.stores,
      };

      setMemoryHistory(prev => {
        const updated = [newDataPoint, ...prev];
        // Keep only data within time range
        const filtered = updated.filter(point => 
          currentTime - point.timestamp <= timeRange
        );
        return filtered.slice(0, 100); // Limit to 100 points max
      });
    }
  }, [memoryData, timeRange]);

  // Time scale for trend charts
  const timeScale = useMemo(() => {
    if (!memoryHistory.length) return null;
    
    const now = Date.now();
    return d3.scaleTime()
      .domain([now - timeRange, now])
      .range([0, chartDimensions.width]);
  }, [memoryHistory, timeRange, chartDimensions.width]);

  // Color scales
  const memoryColorScale = useMemo(() => {
    return d3.scaleLinear<string>()
      .domain([0, CHART_CONFIG.thresholds.memory.warning, CHART_CONFIG.thresholds.memory.critical, 1])
      .range(['#66bb6a', '#4ecdc4', '#ffa726', '#ff6b6b']);
  }, []);

  const storeColorScale = useMemo(() => {
    return d3.scaleOrdinal<string>()
      .domain(['sdr', 'zce', 'cache'])
      .range([
        CHART_CONFIG.colors.stores.sdr,
        CHART_CONFIG.colors.stores.zce,
        CHART_CONFIG.colors.stores.cache,
      ]);
  }, []);

  // Event handlers
  const handleMouseOver = useCallback((data: any, event: MouseEvent) => {
    if (!interactive) return;
    
    setHoveredData(data);
    
    if (svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      setTooltip({
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        content: JSON.stringify(data, null, 2),
      });
    }
  }, [interactive]);

  const handleMouseOut = useCallback(() => {
    if (!interactive) return;
    setHoveredData(null);
    setTooltip(null);
  }, [interactive]);

  // Render memory usage section
  const renderUsageSection = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, sectionHeight: number) => {
    const usageData = memoryData.usage;
    
    // Usage gauge
    const gaugeRadius = Math.min(sectionHeight, chartDimensions.width / 4) * 0.4;
    const gaugeCenterX = gaugeRadius + 20;
    const gaugeCenterY = sectionHeight / 2;
    
    // Background arc
    const arc = d3.arc()
      .innerRadius(gaugeRadius * 0.7)
      .outerRadius(gaugeRadius)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2);

    g.append('path')
      .attr('d', arc as any)
      .attr('transform', `translate(${gaugeCenterX}, ${gaugeCenterY})`)
      .attr('fill', theme === 'dark' ? '#374151' : '#e5e7eb')
      .attr('stroke', 'none');

    // Usage arc
    const usageArc = d3.arc()
      .innerRadius(gaugeRadius * 0.7)
      .outerRadius(gaugeRadius)
      .startAngle(-Math.PI / 2)
      .endAngle(-Math.PI / 2 + Math.PI * usageData.percentage);

    const usagePath = g.append('path')
      .attr('transform', `translate(${gaugeCenterX}, ${gaugeCenterY})`)
      .attr('fill', memoryColorScale(usageData.percentage))
      .attr('stroke', 'none')
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('mouseover', interactive ? (event) => {
        handleMouseOver({
          type: 'memory-usage',
          total: `${(usageData.total / (1024 * 1024 * 1024)).toFixed(2)} GB`,
          used: `${(usageData.used / (1024 * 1024 * 1024)).toFixed(2)} GB`,
          available: `${(usageData.available / (1024 * 1024 * 1024)).toFixed(2)} GB`,
          percentage: `${(usageData.percentage * 100).toFixed(1)}%`,
        }, event);
      } : null)
      .on('mouseout', interactive ? handleMouseOut : null);

    if (enableAnimations) {
      usagePath
        .attr('d', d3.arc()
          .innerRadius(gaugeRadius * 0.7)
          .outerRadius(gaugeRadius)
          .startAngle(-Math.PI / 2)
          .endAngle(-Math.PI / 2) as any)
        .transition()
        .duration(CHART_CONFIG.animation.duration)
        .ease(CHART_CONFIG.animation.ease)
        .attr('d', usageArc as any);
    } else {
      usagePath.attr('d', usageArc as any);
    }

    // Usage percentage text
    g.append('text')
      .attr('x', gaugeCenterX)
      .attr('y', gaugeCenterY)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text(`${(usageData.percentage * 100).toFixed(1)}%`);

    // Memory breakdown bars
    if (showBreakdown) {
      const barWidth = 20;
      const barHeight = sectionHeight * 0.8;
      const barX = gaugeCenterX + gaugeRadius + 40;
      const barY = sectionHeight * 0.1;

      // Total memory bar background
      g.append('rect')
        .attr('x', barX)
        .attr('y', barY)
        .attr('width', barWidth)
        .attr('height', barHeight)
        .attr('fill', theme === 'dark' ? '#374151' : '#e5e7eb')
        .attr('stroke', theme === 'dark' ? '#6b7280' : '#9ca3af')
        .attr('rx', 4);

      // Used memory bar
      const usedHeight = barHeight * usageData.percentage;
      g.append('rect')
        .attr('x', barX)
        .attr('y', barY + barHeight - usedHeight)
        .attr('width', barWidth)
        .attr('height', enableAnimations ? 0 : usedHeight)
        .attr('fill', memoryColorScale(usageData.percentage))
        .attr('rx', 4);

      if (enableAnimations) {
        g.select('rect:last-child')
          .transition()
          .duration(CHART_CONFIG.animation.duration)
          .ease(d3.easeElasticOut)
          .attr('height', usedHeight);
      }

      // Labels
      g.append('text')
        .attr('x', barX + barWidth + 10)
        .attr('y', barY + 15)
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .text(`Total: ${(usageData.total / (1024 * 1024 * 1024)).toFixed(2)} GB`);

      g.append('text')
        .attr('x', barX + barWidth + 10)
        .attr('y', barY + 35)
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .text(`Used: ${(usageData.used / (1024 * 1024 * 1024)).toFixed(2)} GB`);
    }

    // Trend chart
    if (showTrends && memoryHistory.length > 1 && timeScale) {
      const trendX = chartDimensions.width * 0.6;
      const trendWidth = chartDimensions.width * 0.35;
      const trendHeight = sectionHeight * 0.6;

      const yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([trendHeight, 0]);

      const line = d3.line<MemoryTimePoint>()
        .x(d => timeScale(d.timestamp) - (timeScale.range()[0]))
        .y(d => yScale(d.usage))
        .curve(d3.curveMonotoneX);

      // Trend background
      g.append('rect')
        .attr('x', trendX)
        .attr('y', sectionHeight * 0.2)
        .attr('width', trendWidth)
        .attr('height', trendHeight)
        .attr('fill', theme === 'dark' ? '#1e293b' : '#f8fafc')
        .attr('stroke', theme === 'dark' ? '#374151' : '#e5e7eb')
        .attr('rx', 4);

      // Trend line
      const trendGroup = g.append('g')
        .attr('transform', `translate(${trendX}, ${sectionHeight * 0.2})`);

      trendGroup.append('path')
        .datum(memoryHistory.slice(0, 20))
        .attr('fill', 'none')
        .attr('stroke', CHART_CONFIG.colors.usage)
        .attr('stroke-width', 2)
        .attr('d', line);

      // Trend points
      trendGroup.selectAll('.trend-point')
        .data(memoryHistory.slice(0, 10))
        .enter()
        .append('circle')
        .attr('class', 'trend-point')
        .attr('cx', d => timeScale(d.timestamp) - (timeScale.range()[0]))
        .attr('cy', d => yScale(d.usage))
        .attr('r', 3)
        .attr('fill', CHART_CONFIG.colors.usage)
        .attr('stroke', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('stroke-width', 1);
    }
  }, [memoryData, chartDimensions, theme, enableAnimations, interactive, showBreakdown, showTrends, memoryHistory, timeScale, memoryColorScale, handleMouseOver, handleMouseOut]);

  // Render performance section
  const renderPerformanceSection = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, sectionHeight: number) => {
    const performance = memoryData.performance;
    const metrics = [
      { name: 'Latency', value: performance.latency, unit: 'ms', color: CHART_CONFIG.colors.performance.latency, max: 1000 },
      { name: 'Throughput', value: performance.throughput, unit: 'ops/s', color: CHART_CONFIG.colors.performance.throughput, max: 10000 },
      { name: 'Error Rate', value: performance.errorRate * 100, unit: '%', color: CHART_CONFIG.colors.performance.errorRate, max: 10 },
      { name: 'Uptime', value: performance.uptime * 100, unit: '%', color: CHART_CONFIG.colors.performance.uptime, max: 100 },
    ];

    const barHeight = (sectionHeight - 40) / metrics.length;
    const barMaxWidth = chartDimensions.width * 0.6;

    metrics.forEach((metric, index) => {
      const y = index * barHeight + 10;
      const barWidth = (metric.value / metric.max) * barMaxWidth;

      // Background bar
      g.append('rect')
        .attr('x', 120)
        .attr('y', y)
        .attr('width', barMaxWidth)
        .attr('height', barHeight * 0.6)
        .attr('fill', theme === 'dark' ? '#374151' : '#e5e7eb')
        .attr('rx', 4);

      // Value bar
      const valueBar = g.append('rect')
        .attr('x', 120)
        .attr('y', y)
        .attr('width', enableAnimations ? 0 : barWidth)
        .attr('height', barHeight * 0.6)
        .attr('fill', metric.color)
        .attr('rx', 4)
        .style('cursor', interactive ? 'pointer' : 'default')
        .on('mouseover', interactive ? (event) => {
          handleMouseOver({
            type: 'performance',
            metric: metric.name,
            value: metric.value,
            unit: metric.unit,
          }, event);
        } : null)
        .on('mouseout', interactive ? handleMouseOut : null);

      if (enableAnimations) {
        valueBar
          .transition()
          .duration(CHART_CONFIG.animation.duration)
          .ease(d3.easeElasticOut)
          .attr('width', barWidth);
      }

      // Metric label
      g.append('text')
        .attr('x', 10)
        .attr('y', y + (barHeight * 0.6) / 2)
        .attr('dy', '0.35em')
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .text(metric.name);

      // Value label
      g.append('text')
        .attr('x', 120 + barMaxWidth + 10)
        .attr('y', y + (barHeight * 0.6) / 2)
        .attr('dy', '0.35em')
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .text(`${metric.value.toFixed(1)} ${metric.unit}`);
    });
  }, [memoryData, chartDimensions, theme, enableAnimations, interactive, handleMouseOver, handleMouseOut]);

  // Render memory stores section
  const renderStoresSection = useCallback((g: d3.Selection<SVGGElement, unknown, null, undefined>, sectionHeight: number) => {
    const stores = memoryData.stores;
    if (!stores.length) return;

    const radius = Math.min(sectionHeight, chartDimensions.width / 3) * 0.3;
    const centerX = chartDimensions.width / 2;
    const centerY = sectionHeight / 2;

    // Pie chart for store utilization
    const pie = d3.pie<MemoryStore>()
      .value(d => d.size)
      .sort(null);

    const arc = d3.arc<d3.PieArcDatum<MemoryStore>>()
      .innerRadius(radius * 0.5)
      .outerRadius(radius);

    const arcs = g.selectAll('.arc')
      .data(pie(stores))
      .enter()
      .append('g')
      .attr('class', 'arc')
      .attr('transform', `translate(${centerX}, ${centerY})`);

    arcs.append('path')
      .attr('d', arc as any)
      .attr('fill', d => storeColorScale(d.data.type))
      .attr('stroke', theme === 'dark' ? '#374151' : '#ffffff')
      .attr('stroke-width', 2)
      .style('cursor', interactive ? 'pointer' : 'default')
      .on('mouseover', interactive ? (event, d) => {
        handleMouseOver({
          type: 'memory-store',
          name: d.data.name,
          storeType: d.data.type,
          size: `${(d.data.size / (1024 * 1024)).toFixed(2)} MB`,
          utilization: `${(d.data.utilization * 100).toFixed(1)}%`,
          accessCount: d.data.accessCount,
        }, event);
      } : null)
      .on('mouseout', interactive ? handleMouseOut : null);

    // Store labels
    arcs.append('text')
      .attr('transform', d => `translate(${(arc.centroid(d) as [number, number])[0]}, ${(arc.centroid(d) as [number, number])[1]})`)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '11px')
      .attr('font-weight', 'bold')
      .text(d => d.data.type.toUpperCase());

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${centerX + radius + 40}, ${centerY - stores.length * 10})`);

    stores.forEach((store, index) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${index * 20})`);

      legendItem.append('rect')
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', storeColorScale(store.type))
        .attr('rx', 2);

      legendItem.append('text')
        .attr('x', 20)
        .attr('y', 6)
        .attr('dy', '0.35em')
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '12px')
        .text(`${store.name} (${(store.utilization * 100).toFixed(1)}%)`);
    });
  }, [memoryData, chartDimensions, theme, interactive, storeColorScale, handleMouseOver, handleMouseOut]);

  // Main D3 visualization effect
  useEffect(() => {
    if (!svgRef.current || !memoryData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group with margins
    const g = svg
      .append('g')
      .attr('transform', `translate(${CHART_CONFIG.margin.left},${CHART_CONFIG.margin.top})`);

    // Calculate section heights
    const usageHeight = chartDimensions.height * CHART_CONFIG.sections.usage.height;
    const performanceHeight = chartDimensions.height * CHART_CONFIG.sections.performance.height;
    const storesHeight = chartDimensions.height * CHART_CONFIG.sections.stores.height;

    // Section backgrounds and dividers
    const sections = [
      { name: 'usage', height: usageHeight, yOffset: 0, title: CHART_CONFIG.sections.usage.title },
      { name: 'performance', height: performanceHeight, yOffset: usageHeight, title: CHART_CONFIG.sections.performance.title },
      { name: 'stores', height: storesHeight, yOffset: usageHeight + performanceHeight, title: CHART_CONFIG.sections.stores.title },
    ];

    sections.forEach((section, index) => {
      // Section background
      g.append('rect')
        .attr('x', -10)
        .attr('y', section.yOffset - 5)
        .attr('width', chartDimensions.width + 20)
        .attr('height', section.height + 10)
        .attr('fill', index % 2 === 0 ? 
          (theme === 'dark' ? '#1e293b' : '#f8fafc') : 
          (theme === 'dark' ? '#0f172a' : '#ffffff'))
        .attr('stroke', theme === 'dark' ? '#374151' : '#e5e7eb')
        .attr('rx', 8);

      // Section title
      g.append('text')
        .attr('x', 10)
        .attr('y', section.yOffset + 20)
        .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(section.title);

      // Create section group
      const sectionGroup = g.append('g')
        .attr('transform', `translate(0, ${section.yOffset + 30})`);

      // Render section content
      if (section.name === 'usage') {
        renderUsageSection(sectionGroup, section.height - 30);
      } else if (section.name === 'performance') {
        renderPerformanceSection(sectionGroup, section.height - 30);
      } else if (section.name === 'stores') {
        renderStoresSection(sectionGroup, section.height - 30);
      }
    });

    // Main title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', CHART_CONFIG.margin.top / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', theme === 'dark' ? '#ffffff' : '#374151')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text('Memory System Overview');

  }, [memoryData, chartDimensions, theme, width, renderUsageSection, renderPerformanceSection, renderStoresSection]);

  if (!memoryData) {
    return (
      <div 
        className={`flex items-center justify-center ${className}`}
        style={{
          width,
          height,
          background: theme === 'dark' ? '#1e293b' : '#f8fafc',
          borderRadius: '8px',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          color: theme === 'dark' ? '#9ca3af' : '#6b7280',
        }}
      >
        No memory data available
      </div>
    );
  }

  return (
    <div className={`relative ${className}`} style={{ width, height }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          background: theme === 'dark' ? '#0f172a' : '#f8fafc',
          borderRadius: '8px',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        }}
      />
      
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x + 10,
            top: tooltip.y - 10,
            background: theme === 'dark' ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.95)',
            color: theme === 'dark' ? '#ffffff' : '#374151',
            padding: '8px 10px',
            borderRadius: '4px',
            fontSize: '12px',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
            whiteSpace: 'pre-line',
            pointerEvents: 'none',
            zIndex: 1000,
            maxWidth: '200px',
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
};

export default MemorySystemChart;