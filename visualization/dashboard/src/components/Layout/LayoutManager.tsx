import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Layout, Layouts } from 'react-grid-layout';
import GridLayout, { LayoutItem } from './GridLayout';
import { useAppDispatch, useAppSelector } from '../../stores/hooks';
import { 
  setLayout, 
  saveLayoutPreset, 
  loadLayoutPreset, 
  resetLayout,
  selectCurrentLayout,
  selectLayoutPresets,
  selectLayoutSettings 
} from '../../stores/slices/layoutSlice';

export interface LayoutPreset {
  id: string;
  name: string;
  description: string;
  layout: Layouts;
  items: LayoutItem[];
  category: 'cognitive' | 'neural' | 'knowledge' | 'memory' | 'overview' | 'custom';
  tags: string[];
  createdAt: string;
  updatedAt: string;
  isDefault?: boolean;
}

export interface LayoutManagerProps {
  items: LayoutItem[];
  onItemsChange?: (items: LayoutItem[]) => void;
  allowPresets?: boolean;
  allowCustomization?: boolean;
  category?: string;
  className?: string;
  style?: React.CSSProperties;
}

const defaultPresets: LayoutPreset[] = [
  {
    id: 'cognitive-analysis',
    name: 'Cognitive Analysis',
    description: 'Optimized layout for pattern recognition and cognitive monitoring',
    layout: {
      lg: [
        { i: 'pattern-recognition', x: 0, y: 0, w: 8, h: 4 },
        { i: 'cognitive-load', x: 8, y: 0, w: 4, h: 2 },
        { i: 'attention-heatmap', x: 8, y: 2, w: 4, h: 2 },
        { i: 'inhibition-control', x: 0, y: 4, w: 6, h: 3 },
        { i: 'memory-trace', x: 6, y: 4, w: 6, h: 3 }
      ],
      md: [
        { i: 'pattern-recognition', x: 0, y: 0, w: 6, h: 4 },
        { i: 'cognitive-load', x: 6, y: 0, w: 4, h: 2 },
        { i: 'attention-heatmap', x: 6, y: 2, w: 4, h: 2 },
        { i: 'inhibition-control', x: 0, y: 4, w: 5, h: 3 },
        { i: 'memory-trace', x: 5, y: 4, w: 5, h: 3 }
      ]
    },
    items: [],
    category: 'cognitive',
    tags: ['patterns', 'attention', 'inhibition'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    isDefault: true
  },
  {
    id: 'neural-monitoring',
    name: 'Neural Activity Monitor',
    description: 'Real-time neural activity visualization with emphasis on heatmaps',
    layout: {
      lg: [
        { i: 'neural-heatmap', x: 0, y: 0, w: 12, h: 5 },
        { i: 'spike-patterns', x: 0, y: 5, w: 6, h: 3 },
        { i: 'connection-strength', x: 6, y: 5, w: 6, h: 3 },
        { i: 'activity-timeline', x: 0, y: 8, w: 12, h: 2 }
      ],
      md: [
        { i: 'neural-heatmap', x: 0, y: 0, w: 10, h: 5 },
        { i: 'spike-patterns', x: 0, y: 5, w: 5, h: 3 },
        { i: 'connection-strength', x: 5, y: 5, w: 5, h: 3 },
        { i: 'activity-timeline', x: 0, y: 8, w: 10, h: 2 }
      ]
    },
    items: [],
    category: 'neural',
    tags: ['neural', 'activity', 'realtime'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    isDefault: true
  },
  {
    id: 'knowledge-graph',
    name: 'Knowledge Graph Explorer',
    description: '3D knowledge graph visualization with detailed node inspection',
    layout: {
      lg: [
        { i: '3d-knowledge-graph', x: 0, y: 0, w: 9, h: 8 },
        { i: 'node-inspector', x: 9, y: 0, w: 3, h: 4 },
        { i: 'relation-explorer', x: 9, y: 4, w: 3, h: 4 },
        { i: 'graph-metrics', x: 0, y: 8, w: 6, h: 2 },
        { i: 'search-panel', x: 6, y: 8, w: 6, h: 2 }
      ],
      md: [
        { i: '3d-knowledge-graph', x: 0, y: 0, w: 7, h: 6 },
        { i: 'node-inspector', x: 7, y: 0, w: 3, h: 3 },
        { i: 'relation-explorer', x: 7, y: 3, w: 3, h: 3 },
        { i: 'graph-metrics', x: 0, y: 6, w: 5, h: 2 },
        { i: 'search-panel', x: 5, y: 6, w: 5, h: 2 }
      ]
    },
    items: [],
    category: 'knowledge',
    tags: ['3d', 'graph', 'exploration'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    isDefault: true
  },
  {
    id: 'memory-analytics',
    name: 'Memory Performance',
    description: 'Memory system analytics with performance optimization focus',
    layout: {
      lg: [
        { i: 'memory-usage', x: 0, y: 0, w: 6, h: 3 },
        { i: 'cache-efficiency', x: 6, y: 0, w: 6, h: 3 },
        { i: 'gc-metrics', x: 0, y: 3, w: 4, h: 3 },
        { i: 'allocation-timeline', x: 4, y: 3, w: 8, h: 3 },
        { i: 'memory-leaks', x: 0, y: 6, w: 12, h: 2 }
      ],
      md: [
        { i: 'memory-usage', x: 0, y: 0, w: 5, h: 3 },
        { i: 'cache-efficiency', x: 5, y: 0, w: 5, h: 3 },
        { i: 'gc-metrics', x: 0, y: 3, w: 3, h: 3 },
        { i: 'allocation-timeline', x: 3, y: 3, w: 7, h: 3 },
        { i: 'memory-leaks', x: 0, y: 6, w: 10, h: 2 }
      ]
    },
    items: [],
    category: 'memory',
    tags: ['memory', 'performance', 'optimization'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    isDefault: true
  },
  {
    id: 'system-overview',
    name: 'System Overview',
    description: 'Comprehensive system monitoring with all key components',
    layout: {
      lg: [
        { i: 'system-status', x: 0, y: 0, w: 3, h: 2 },
        { i: 'performance-metrics', x: 3, y: 0, w: 3, h: 2 },
        { i: 'resource-usage', x: 6, y: 0, w: 3, h: 2 },
        { i: 'alerts-panel', x: 9, y: 0, w: 3, h: 2 },
        { i: 'neural-overview', x: 0, y: 2, w: 6, h: 4 },
        { i: 'cognitive-overview', x: 6, y: 2, w: 6, h: 4 },
        { i: 'knowledge-summary', x: 0, y: 6, w: 4, h: 3 },
        { i: 'memory-summary', x: 4, y: 6, w: 4, h: 3 },
        { i: 'activity-log', x: 8, y: 6, w: 4, h: 3 }
      ],
      md: [
        { i: 'system-status', x: 0, y: 0, w: 2, h: 2 },
        { i: 'performance-metrics', x: 2, y: 0, w: 3, h: 2 },
        { i: 'resource-usage', x: 5, y: 0, w: 3, h: 2 },
        { i: 'alerts-panel', x: 8, y: 0, w: 2, h: 2 },
        { i: 'neural-overview', x: 0, y: 2, w: 5, h: 4 },
        { i: 'cognitive-overview', x: 5, y: 2, w: 5, h: 4 },
        { i: 'knowledge-summary', x: 0, y: 6, w: 3, h: 3 },
        { i: 'memory-summary', x: 3, y: 6, w: 3, h: 3 },
        { i: 'activity-log', x: 6, y: 6, w: 4, h: 3 }
      ]
    },
    items: [],
    category: 'overview',
    tags: ['overview', 'monitoring', 'dashboard'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    isDefault: true
  }
];

export const LayoutManager: React.FC<LayoutManagerProps> = ({
  items,
  onItemsChange,
  allowPresets = true,
  allowCustomization = true,
  category,
  className = '',
  style = {}
}) => {
  const dispatch = useAppDispatch();
  const currentLayout = useAppSelector(selectCurrentLayout);
  const savedPresets = useAppSelector(selectLayoutPresets);
  const layoutSettings = useAppSelector(selectLayoutSettings);

  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [showPresetDialog, setShowPresetDialog] = useState(false);
  const [customPresetName, setCustomPresetName] = useState('');
  const [customPresetDescription, setCustomPresetDescription] = useState('');

  const availablePresets = useMemo(() => {
    const allPresets = [...defaultPresets, ...savedPresets];
    return category ? 
      allPresets.filter(preset => preset.category === category) : 
      allPresets;
  }, [savedPresets, category]);

  const handleLayoutChange = useCallback((layout: Layout[], layouts: Layouts) => {
    dispatch(setLayout({ layouts, items }));
  }, [dispatch, items]);

  const handleLoadPreset = useCallback((presetId: string) => {
    const preset = availablePresets.find(p => p.id === presetId);
    if (preset) {
      dispatch(loadLayoutPreset(presetId));
      setActivePreset(presetId);
      
      // Update items if the preset has custom items
      if (preset.items.length > 0 && onItemsChange) {
        onItemsChange(preset.items);
      }
    }
  }, [availablePresets, dispatch, onItemsChange]);

  const handleSavePreset = useCallback(() => {
    if (!customPresetName.trim()) return;

    const newPreset: LayoutPreset = {
      id: `custom-${Date.now()}`,
      name: customPresetName,
      description: customPresetDescription,
      layout: currentLayout,
      items,
      category: (category as any) || 'custom',
      tags: ['custom'],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    dispatch(saveLayoutPreset(newPreset));
    setShowPresetDialog(false);
    setCustomPresetName('');
    setCustomPresetDescription('');
  }, [customPresetName, customPresetDescription, currentLayout, items, category, dispatch]);

  const handleResetLayout = useCallback(() => {
    dispatch(resetLayout());
    setActivePreset(null);
  }, [dispatch]);

  const handleExportLayout = useCallback(() => {
    const exportData = {
      layout: currentLayout,
      items,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llmkg-layout-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [currentLayout, items]);

  const handleImportLayout = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importData = JSON.parse(e.target?.result as string);
        if (importData.layout && importData.items) {
          dispatch(setLayout({ 
            layouts: importData.layout, 
            items: importData.items 
          }));
          if (onItemsChange) {
            onItemsChange(importData.items);
          }
          setActivePreset(null);
        }
      } catch (error) {
        console.error('Failed to import layout:', error);
      }
    };
    reader.readAsText(file);
  }, [dispatch, onItemsChange]);

  return (
    <div className={`layout-manager ${className}`} style={style}>
      {allowPresets && (
        <div className="layout-controls" style={{ marginBottom: '20px' }}>
          <div className="preset-controls" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
            <select
              value={activePreset || ''}
              onChange={(e) => e.target.value && handleLoadPreset(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="">Select Layout Preset</option>
              {availablePresets.map(preset => (
                <option key={preset.id} value={preset.id}>
                  {preset.name} {preset.isDefault ? '(Default)' : ''}
                </option>
              ))}
            </select>

            {allowCustomization && (
              <>
                <button
                  onClick={() => setShowPresetDialog(true)}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Save Preset
                </button>

                <button
                  onClick={handleResetLayout}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Reset
                </button>

                <button
                  onClick={handleExportLayout}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#28a745',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Export
                </button>

                <label
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#17a2b8',
                    color: 'white',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Import
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImportLayout}
                    style={{ display: 'none' }}
                  />
                </label>
              </>
            )}
          </div>

          {activePreset && (
            <div className="active-preset-info" style={{ marginTop: '10px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '6px' }}>
              {(() => {
                const preset = availablePresets.find(p => p.id === activePreset);
                return preset ? (
                  <div>
                    <strong>{preset.name}</strong>
                    <p style={{ margin: '5px 0 0 0', fontSize: '14px', color: '#666' }}>
                      {preset.description}
                    </p>
                    <div style={{ marginTop: '5px' }}>
                      {preset.tags.map(tag => (
                        <span
                          key={tag}
                          style={{
                            display: 'inline-block',
                            padding: '2px 8px',
                            margin: '2px',
                            backgroundColor: '#e9ecef',
                            borderRadius: '12px',
                            fontSize: '12px',
                            color: '#495057'
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null;
              })()}
            </div>
          )}
        </div>
      )}

      <GridLayout
        items={items}
        onLayoutChange={handleLayoutChange}
        layouts={currentLayout}
        isDraggable={layoutSettings.isDraggable}
        isResizable={layoutSettings.isResizable}
        compactType={layoutSettings.compactType}
        preventCollision={layoutSettings.preventCollision}
        margin={layoutSettings.margin}
        containerPadding={layoutSettings.containerPadding}
        rowHeight={layoutSettings.rowHeight}
      />

      {showPresetDialog && (
        <div 
          className="preset-dialog-overlay"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000
          }}
          onClick={() => setShowPresetDialog(false)}
        >
          <div 
            className="preset-dialog"
            style={{
              backgroundColor: 'white',
              padding: '30px',
              borderRadius: '12px',
              minWidth: '400px',
              boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: '0 0 20px 0' }}>Save Layout Preset</h3>
            
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Preset Name:
              </label>
              <input
                type="text"
                value={customPresetName}
                onChange={(e) => setCustomPresetName(e.target.value)}
                placeholder="Enter preset name"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Description:
              </label>
              <textarea
                value={customPresetDescription}
                onChange={(e) => setCustomPresetDescription(e.target.value)}
                placeholder="Enter preset description (optional)"
                rows={3}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  fontSize: '14px',
                  resize: 'vertical'
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowPresetDialog(false)}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSavePreset}
                disabled={!customPresetName.trim()}
                style={{
                  padding: '10px 20px',
                  backgroundColor: customPresetName.trim() ? '#007bff' : '#ccc',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: customPresetName.trim() ? 'pointer' : 'not-allowed'
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LayoutManager;