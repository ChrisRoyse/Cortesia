import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { InhibitionExcitationBalance as IEBalance } from '../types/cognitive';

interface InhibitionExcitationBalanceProps {
  balanceData: IEBalance[];
  currentBalance: IEBalance;
  className?: string;
}

export function InhibitionExcitationBalance({ 
  balanceData, 
  currentBalance, 
  className = '' 
}: InhibitionExcitationBalanceProps) {
  const chartRef = useRef<SVGSVGElement>(null);
  const gaugeRef = useRef<SVGSVGElement>(null);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);

  // Time series chart
  useEffect(() => {
    if (!chartRef.current || balanceData.length === 0) return;

    const margin = { top: 20, right: 80, bottom: 40, left: 60 };
    const width = 700 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(balanceData, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([height, 0]);

    // Gradient for balance area
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'balance-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#10b981')
      .attr('stop-opacity', 0.8);

    gradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#6b7280')
      .attr('stop-opacity', 0.5);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#ef4444')
      .attr('stop-opacity', 0.8);

    // Areas
    const excitationArea = d3.area<IEBalance>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(yScale(0))
      .y1(d => yScale(d.balance))
      .curve(d3.curveMonotoneX);

    const inhibitionArea = d3.area<IEBalance>()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(yScale(0))
      .y1(d => yScale(d.balance))
      .curve(d3.curveMonotoneX);

    // Lines
    const balanceLine = d3.line<IEBalance>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.balance))
      .curve(d3.curveMonotoneX);

    const excitationLine = d3.line<IEBalance>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.excitation.total / (d.excitation.total + d.inhibition.total) * 2 - 1))
      .curve(d3.curveMonotoneX);

    const inhibitionLine = d3.line<IEBalance>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(-d.inhibition.total / (d.excitation.total + d.inhibition.total) * 2 + 1))
      .curve(d3.curveMonotoneX);

    // Draw optimal range
    const optimalRange = currentBalance.optimalRange;
    g.append('rect')
      .attr('x', 0)
      .attr('y', yScale(optimalRange[1]))
      .attr('width', width)
      .attr('height', yScale(optimalRange[0]) - yScale(optimalRange[1]))
      .attr('fill', '#10b981')
      .attr('opacity', 0.1);

    // Draw areas
    g.append('path')
      .datum(balanceData)
      .attr('fill', 'url(#balance-gradient)')
      .attr('opacity', 0.3)
      .attr('d', excitationArea);

    // Draw lines
    g.append('path')
      .datum(balanceData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 3)
      .attr('d', balanceLine);

    g.append('path')
      .datum(balanceData)
      .attr('fill', 'none')
      .attr('stroke', '#10b981')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('d', excitationLine);

    g.append('path')
      .datum(balanceData)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('d', inhibitionLine);

    // Zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2');

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')))
      .append('text')
      .attr('x', width / 2)
      .attr('y', 35)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Time');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => 
        d === 1 ? 'Full Excitation' : 
        d === -1 ? 'Full Inhibition' : 
        d === 0 ? 'Balanced' : d.toString()
      ))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height / 2)
      .attr('fill', '#9ca3af')
      .style('text-anchor', 'middle')
      .text('Balance');

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${width - 120}, 10)`);

    const legendItems = [
      { color: '#3b82f6', label: 'Balance', dash: false },
      { color: '#10b981', label: 'Excitation', dash: true },
      { color: '#ef4444', label: 'Inhibition', dash: true }
    ];

    legendItems.forEach((item, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendRow.append('line')
        .attr('x1', 0)
        .attr('x2', 20)
        .attr('stroke', item.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', item.dash ? '5,5' : '');

      legendRow.append('text')
        .attr('x', 25)
        .attr('y', 5)
        .attr('font-size', '12px')
        .attr('fill', '#9ca3af')
        .text(item.label);
    });
  }, [balanceData, currentBalance]);

  // Balance gauge
  useEffect(() => {
    if (!gaugeRef.current) return;

    const width = 300;
    const height = 200;
    const margin = 40;

    const svg = d3.select(gaugeRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height - margin})`);

    // Gauge arc
    const arc = d3.arc()
      .innerRadius(60)
      .outerRadius(80)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2);

    // Background arc
    g.append('path')
      .attr('d', arc as any)
      .attr('fill', '#374151');

    // Gradient for gauge
    const gaugeGradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'gauge-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '100%')
      .attr('y2', '0%');

    gaugeGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#ef4444');

    gaugeGradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#6b7280');

    gaugeGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#10b981');

    // Value arc
    const valueArc = d3.arc()
      .innerRadius(60)
      .outerRadius(80)
      .startAngle(-Math.PI / 2)
      .endAngle((-Math.PI / 2) + (Math.PI * (currentBalance.balance + 1) / 2));

    g.append('path')
      .attr('d', valueArc as any)
      .attr('fill', 'url(#gauge-gradient)');

    // Needle
    const needleAngle = (-Math.PI / 2) + (Math.PI * (currentBalance.balance + 1) / 2);
    g.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', 70 * Math.cos(needleAngle))
      .attr('y2', 70 * Math.sin(needleAngle))
      .attr('stroke', 'white')
      .attr('stroke-width', 3);

    g.append('circle')
      .attr('r', 5)
      .attr('fill', 'white');

    // Labels
    g.append('text')
      .attr('x', -80)
      .attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#ef4444')
      .text('Inhibition');

    g.append('text')
      .attr('x', 80)
      .attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#10b981')
      .text('Excitation');

    g.append('text')
      .attr('x', 0)
      .attr('y', -90)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#6b7280')
      .text('Balanced');

    // Value text
    g.append('text')
      .attr('x', 0)
      .attr('y', 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('fill', 'white')
      .text(currentBalance.balance.toFixed(2));
  }, [currentBalance]);

  const getBalanceStatus = (balance: number): { status: string; color: string; description: string } => {
    if (balance > 0.5) {
      return {
        status: 'High Excitation',
        color: 'text-green-400',
        description: 'System is in a highly excited state, may lead to instability'
      };
    } else if (balance > 0.2) {
      return {
        status: 'Moderate Excitation',
        color: 'text-green-300',
        description: 'Healthy excitation level for active processing'
      };
    } else if (balance > -0.2) {
      return {
        status: 'Balanced',
        color: 'text-gray-300',
        description: 'Optimal balance between excitation and inhibition'
      };
    } else if (balance > -0.5) {
      return {
        status: 'Moderate Inhibition',
        color: 'text-red-300',
        description: 'Controlled inhibition for stability'
      };
    } else {
      return {
        status: 'High Inhibition',
        color: 'text-red-400',
        description: 'System is heavily inhibited, may affect responsiveness'
      };
    }
  };

  const balanceStatus = getBalanceStatus(currentBalance.balance);

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <h3 className="text-xl font-semibold text-white mb-4">Inhibition/Excitation Balance</h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Balance Gauge */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Current Balance</h4>
          <svg 
            ref={gaugeRef}
            width="300"
            height="200"
            className="w-full h-auto"
            viewBox="0 0 300 200"
            preserveAspectRatio="xMidYMid meet"
          />
          <div className="mt-4 text-center">
            <div className={`text-lg font-medium ${balanceStatus.color}`}>
              {balanceStatus.status}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {balanceStatus.description}
            </div>
          </div>
        </div>

        {/* Time Series Chart */}
        <div className="lg:col-span-2 bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Balance History</h4>
          <svg 
            ref={chartRef}
            width="700"
            height="300"
            className="w-full h-auto"
            viewBox="0 0 700 300"
            preserveAspectRatio="xMidYMid meet"
          />
        </div>
      </div>

      {/* Regional Balance */}
      <div className="mt-6 bg-gray-800 rounded p-4">
        <h4 className="text-lg font-medium text-white mb-3">Regional Balance</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(currentBalance.excitation.byRegion).map(([region, excitation]) => {
            const inhibition = currentBalance.inhibition.byRegion[region] || 0;
            const total = excitation + inhibition;
            const regionalBalance = total > 0 ? (excitation - inhibition) / total : 0;
            
            return (
              <div 
                key={region}
                className={`bg-gray-700 rounded p-3 cursor-pointer transition-all ${
                  selectedRegion === region ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedRegion(region === selectedRegion ? null : region)}
              >
                <div className="text-sm text-gray-400 capitalize">{region}</div>
                <div className="mt-2 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                      <div 
                        className="h-full transition-all duration-300"
                        style={{
                          width: `${Math.abs(regionalBalance) * 50 + 50}%`,
                          backgroundColor: regionalBalance > 0 ? '#10b981' : '#ef4444',
                          marginLeft: regionalBalance < 0 ? `${50 - Math.abs(regionalBalance) * 50}%` : '0'
                        }}
                      />
                    </div>
                  </div>
                  <span className="ml-2 text-xs text-white">
                    {(regionalBalance * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-1 flex justify-between text-xs">
                  <span className="text-green-400">E: {excitation.toFixed(1)}</span>
                  <span className="text-red-400">I: {inhibition.toFixed(1)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Patterns */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Excitatory Patterns</h4>
          <div className="space-y-2">
            {currentBalance.excitation.patterns.slice(0, 5).map((pattern, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-sm text-gray-300">{pattern}</span>
                <span className="text-xs text-green-400">Active</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Inhibitory Patterns</h4>
          <div className="space-y-2">
            {currentBalance.inhibition.patterns.slice(0, 5).map((pattern, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-sm text-gray-300">{pattern}</span>
                <span className="text-xs text-red-400">Active</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {Math.abs(currentBalance.balance) > 0.5 && (
        <div className="mt-6 bg-yellow-500/10 border border-yellow-500/20 rounded p-4">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-yellow-500 mt-0.5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <div className="text-yellow-400 font-medium">Balance Alert</div>
              <div className="text-sm text-gray-300 mt-1">
                {currentBalance.balance > 0.5 
                  ? 'System is experiencing high excitation. Consider increasing inhibitory mechanisms to prevent runaway activation.'
                  : 'System is heavily inhibited. Consider reducing inhibitory signals to improve responsiveness.'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}