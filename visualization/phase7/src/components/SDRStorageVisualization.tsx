import React, { useMemo } from 'react';
import { SDRStorage, SDRStorageBlock } from '../types/memory';

interface SDRStorageVisualizationProps {
  storage: SDRStorage;
  className?: string;
}

export function SDRStorageVisualization({ storage, className = '' }: SDRStorageVisualizationProps) {
  const blockColors = useMemo(() => {
    return storage.storageBlocks.map(block => {
      const usage = block.used / block.size;
      const fragmentation = block.fragmented / block.used;
      
      // Color based on usage and fragmentation
      if (fragmentation > 0.3) return '#ef4444'; // Red for high fragmentation
      if (usage > 0.9) return '#f59e0b'; // Orange for high usage
      if (usage > 0.7) return '#eab308'; // Yellow for medium-high usage
      return '#22c55e'; // Green for healthy
    });
  }, [storage.storageBlocks]);

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

  return (
    <div className={`bg-gray-900 rounded-lg p-6 ${className}`}>
      <h3 className="text-xl font-semibold mb-4 text-white">SDR Storage Analysis</h3>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Total SDRs</div>
          <div className="text-2xl font-bold text-white">{storage.totalSDRs.toLocaleString()}</div>
          <div className="text-xs text-gray-500">
            {storage.activeSDRs} active / {storage.archivedSDRs} archived
          </div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Memory Usage</div>
          <div className="text-2xl font-bold text-white">{formatBytes(storage.totalMemoryBytes)}</div>
          <div className="text-xs text-gray-500">
            Compression: {(storage.compressionRatio * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Avg Sparsity</div>
          <div className="text-2xl font-bold text-white">{(storage.averageSparsity * 100).toFixed(1)}%</div>
          <div className="text-xs text-gray-500">Target: 2%</div>
        </div>
        
        <div className="bg-gray-800 rounded p-4">
          <div className="text-gray-400 text-sm">Fragmentation</div>
          <div className="text-2xl font-bold text-white">{(storage.fragmentationLevel * 100).toFixed(1)}%</div>
          <div className={`text-xs ${storage.fragmentationLevel > 0.3 ? 'text-red-400' : 'text-gray-500'}`}>
            {storage.fragmentationLevel > 0.3 ? 'Defrag recommended' : 'Healthy'}
          </div>
        </div>
      </div>

      {/* Storage Blocks Visualization */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium text-white">Storage Blocks</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {storage.storageBlocks.map((block, index) => (
            <StorageBlockVisualization
              key={block.id}
              block={block}
              color={blockColors[index]}
            />
          ))}
        </div>
      </div>

      {/* Fragmentation Heatmap */}
      <div className="mt-6">
        <h4 className="text-lg font-medium text-white mb-3">Fragmentation Heatmap</h4>
        <div className="bg-gray-800 rounded p-4">
          <div className="grid grid-cols-20 gap-1">
            {storage.storageBlocks.map((block, i) => (
              <div
                key={i}
                className="aspect-square rounded"
                style={{
                  backgroundColor: getFragmentationColor(block.fragmented / block.used),
                  opacity: block.used > 0 ? 1 : 0.3
                }}
                title={`Block ${block.id}: ${((block.fragmented / block.used) * 100).toFixed(1)}% fragmented`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

interface StorageBlockVisualizationProps {
  block: SDRStorageBlock;
  color: string;
}

function StorageBlockVisualization({ block, color }: StorageBlockVisualizationProps) {
  const usagePercent = (block.used / block.size) * 100;
  const fragmentedPercent = (block.fragmented / block.size) * 100;
  
  return (
    <div className="bg-gray-800 rounded p-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-gray-300">Block {block.id}</span>
        <span className="text-xs text-gray-500">{block.compressionType}</span>
      </div>
      
      <div className="relative h-8 bg-gray-700 rounded overflow-hidden mb-2">
        <div
          className="absolute h-full transition-all duration-300"
          style={{
            width: `${usagePercent}%`,
            backgroundColor: color
          }}
        />
        <div
          className="absolute h-full bg-red-500 opacity-50"
          style={{
            width: `${fragmentedPercent}%`
          }}
        />
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-gray-500">Used:</span>
          <span className="text-gray-300 ml-1">{usagePercent.toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Patterns:</span>
          <span className="text-gray-300 ml-1">{block.patterns}</span>
        </div>
      </div>
    </div>
  );
}

function getFragmentationColor(fragmentation: number): string {
  if (fragmentation > 0.5) return '#dc2626';
  if (fragmentation > 0.3) return '#f59e0b';
  if (fragmentation > 0.1) return '#eab308';
  return '#22c55e';
}