import React from 'react';
import { CognitiveLayerMemory, WorkingMemoryBuffer } from '../types/memory';

interface CognitiveLayerMemoryProps {
  memory: CognitiveLayerMemory;
  className?: string;
}

export function CognitiveLayerMemoryVisualization({ memory, className = '' }: CognitiveLayerMemoryProps) {
  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  const getUsageColor = (usage: number): string => {
    if (usage > 0.9) return 'bg-red-500';
    if (usage > 0.7) return 'bg-orange-500';
    if (usage > 0.5) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const subcorticalUsage = memory.subcortical.used / memory.subcortical.total;
  const corticalUsage = memory.cortical.used / memory.cortical.total;
  const workingMemoryUsage = memory.workingMemory.used / memory.workingMemory.capacity;

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <h3 className="text-xl font-semibold mb-4 text-white">Cognitive Layer Memory</h3>

      {/* Main Memory Regions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Subcortical */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-2">Subcortical</h4>
          <div className="mb-2">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Usage</span>
              <span className="text-white">{(subcorticalUsage * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${getUsageColor(subcorticalUsage)}`}
                style={{ width: `${subcorticalUsage * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-gray-500 mb-3">
            {formatBytes(memory.subcortical.used)} / {formatBytes(memory.subcortical.total)}
          </div>
          
          {/* Component breakdown */}
          <div className="space-y-2">
            {Object.entries(memory.subcortical.components).map(([component, size]) => (
              <div key={component} className="flex justify-between items-center">
                <span className="text-xs text-gray-400 capitalize">{component}</span>
                <div className="flex items-center">
                  <div className="w-16 bg-gray-700 rounded-full h-1 mr-2">
                    <div 
                      className="h-1 rounded-full bg-blue-500"
                      style={{ width: `${(size / memory.subcortical.used) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-300">{formatBytes(size)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Cortical */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-2">Cortical</h4>
          <div className="mb-2">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Usage</span>
              <span className="text-white">{(corticalUsage * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${getUsageColor(corticalUsage)}`}
                style={{ width: `${corticalUsage * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-gray-500 mb-3">
            {formatBytes(memory.cortical.used)} / {formatBytes(memory.cortical.total)}
          </div>
          
          {/* Region breakdown */}
          <div className="space-y-2">
            {Object.entries(memory.cortical.regions).map(([region, size]) => (
              <div key={region} className="flex justify-between items-center">
                <span className="text-xs text-gray-400 capitalize">{region}</span>
                <div className="flex items-center">
                  <div className="w-16 bg-gray-700 rounded-full h-1 mr-2">
                    <div 
                      className="h-1 rounded-full bg-purple-500"
                      style={{ width: `${(size / memory.cortical.used) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-300">{formatBytes(size)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Working Memory */}
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-2">Working Memory</h4>
          <div className="mb-2">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Usage</span>
              <span className="text-white">{(workingMemoryUsage * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${getUsageColor(workingMemoryUsage)}`}
                style={{ width: `${workingMemoryUsage * 100}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-gray-500 mb-3">
            {formatBytes(memory.workingMemory.used)} / {formatBytes(memory.workingMemory.capacity)}
          </div>
          
          {/* Active buffers */}
          <div className="text-xs text-gray-400 mb-2">
            Active Buffers: {memory.workingMemory.buffers.length}
          </div>
        </div>
      </div>

      {/* Working Memory Buffers Detail */}
      {memory.workingMemory.buffers.length > 0 && (
        <div className="bg-gray-800 rounded p-4">
          <h4 className="text-lg font-medium text-white mb-3">Working Memory Buffers</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {memory.workingMemory.buffers
              .sort((a, b) => b.priority - a.priority)
              .map((buffer) => (
                <WorkingMemoryBufferItem key={buffer.id} buffer={buffer} />
              ))}
          </div>
        </div>
      )}

      {/* Memory Pressure Indicator */}
      <MemoryPressureIndicator 
        subcorticalUsage={subcorticalUsage}
        corticalUsage={corticalUsage}
        workingMemoryUsage={workingMemoryUsage}
      />
    </div>
  );
}

interface WorkingMemoryBufferItemProps {
  buffer: WorkingMemoryBuffer;
}

function WorkingMemoryBufferItem({ buffer }: WorkingMemoryBufferItemProps) {
  const ageColor = buffer.age < 1000 ? 'text-green-400' : 
                   buffer.age < 5000 ? 'text-yellow-400' : 'text-red-400';
  
  return (
    <div className="flex items-center justify-between bg-gray-700 rounded p-2">
      <div className="flex-1">
        <div className="text-sm text-white truncate">{buffer.content}</div>
        <div className="flex items-center space-x-4 text-xs text-gray-400">
          <span>Size: {buffer.size}B</span>
          <span className={ageColor}>Age: {(buffer.age / 1000).toFixed(1)}s</span>
          <span>Access: {buffer.accessCount}</span>
        </div>
      </div>
      <div className="ml-2">
        <div className="text-xs text-gray-500">Priority</div>
        <div className="text-sm font-medium text-white">{buffer.priority.toFixed(2)}</div>
      </div>
    </div>
  );
}

interface MemoryPressureIndicatorProps {
  subcorticalUsage: number;
  corticalUsage: number;
  workingMemoryUsage: number;
}

function MemoryPressureIndicator({ 
  subcorticalUsage, 
  corticalUsage, 
  workingMemoryUsage 
}: MemoryPressureIndicatorProps) {
  const avgUsage = (subcorticalUsage + corticalUsage + workingMemoryUsage) / 3;
  
  let pressureLevel: 'low' | 'medium' | 'high' | 'critical';
  let pressureColor: string;
  let recommendations: string[] = [];

  if (avgUsage > 0.9) {
    pressureLevel = 'critical';
    pressureColor = 'bg-red-500';
    recommendations = [
      'Immediate memory cleanup required',
      'Consider increasing memory allocation',
      'Disable non-essential features'
    ];
  } else if (avgUsage > 0.7) {
    pressureLevel = 'high';
    pressureColor = 'bg-orange-500';
    recommendations = [
      'Monitor memory usage closely',
      'Clear unused caches',
      'Reduce working memory buffer size'
    ];
  } else if (avgUsage > 0.5) {
    pressureLevel = 'medium';
    pressureColor = 'bg-yellow-500';
    recommendations = [
      'Memory usage is moderate',
      'Consider optimizing large buffers'
    ];
  } else {
    pressureLevel = 'low';
    pressureColor = 'bg-green-500';
    recommendations = ['Memory usage is healthy'];
  }

  return (
    <div className="mt-4 bg-gray-800 rounded p-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-lg font-medium text-white">Memory Pressure</h4>
        <div className={`px-3 py-1 rounded text-sm text-white ${pressureColor}`}>
          {pressureLevel.toUpperCase()}
        </div>
      </div>
      <div className="space-y-1">
        {recommendations.map((rec, i) => (
          <div key={i} className="text-sm text-gray-400 flex items-center">
            <span className="mr-2">•</span>
            {rec}
          </div>
        ))}
      </div>
    </div>
  );
}