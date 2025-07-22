/**
 * Master control panel for LLMKG Phase 4 visualization
 * Provides comprehensive control over all visualization aspects
 */

import React, { useState, useEffect, useCallback } from 'react';
import { filteringSystem, FilterGroup, FilterCondition, FilterPreset } from './FilteringSystem';

interface LayerVisibility {
  mcpRequests: boolean;
  cognitivePatterns: boolean;
  memoryOperations: boolean;
  performanceMetrics: boolean;
  connections: boolean;
  particles: boolean;
  heatmaps: boolean;
  timelines: boolean;
}

interface CameraPreset {
  name: string;
  position: [number, number, number];
  target: [number, number, number];
  fov?: number;
}

interface ColorTheme {
  name: string;
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  text: string;
  success: string;
  warning: string;
  error: string;
}

interface PlaybackState {
  isPlaying: boolean;
  speed: number;
  currentTime: Date;
  startTime: Date;
  endTime: Date;
}

interface VisualizationSettings {
  layers: LayerVisibility;
  quality: {
    particleCount: number;
    renderQuality: 'low' | 'medium' | 'high' | 'ultra';
    antiAliasing: boolean;
    shadows: boolean;
    postProcessing: boolean;
  };
  camera: {
    currentPreset?: string;
    smoothTransitions: boolean;
    followMouse: boolean;
    autoRotate: boolean;
  };
  theme: {
    current: string;
    accessibility: {
      highContrast: boolean;
      reducedMotion: boolean;
      colorBlindSafe: boolean;
    };
  };
  playback: PlaybackState;
}

interface VisualizationControlsProps {
  onSettingsChange: (settings: VisualizationSettings) => void;
  onExportRequest: (type: 'screenshot' | 'video' | 'data') => void;
  onDebugToggle: (enabled: boolean) => void;
  onPerformanceMonitor: (enabled: boolean) => void;
  initialSettings?: Partial<VisualizationSettings>;
}

const VisualizationControls: React.FC<VisualizationControlsProps> = ({
  onSettingsChange,
  onExportRequest,
  onDebugToggle,
  onPerformanceMonitor,
  initialSettings = {}
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'layers' | 'filters' | 'quality' | 'camera' | 'theme'>('layers');
  const [settings, setSettings] = useState<VisualizationSettings>({
    layers: {
      mcpRequests: true,
      cognitivePatterns: true,
      memoryOperations: true,
      performanceMetrics: false,
      connections: true,
      particles: true,
      heatmaps: false,
      timelines: false,
      ...initialSettings.layers
    },
    quality: {
      particleCount: 5000,
      renderQuality: 'high',
      antiAliasing: true,
      shadows: true,
      postProcessing: true,
      ...initialSettings.quality
    },
    camera: {
      smoothTransitions: true,
      followMouse: false,
      autoRotate: false,
      ...initialSettings.camera
    },
    theme: {
      current: 'dark',
      accessibility: {
        highContrast: false,
        reducedMotion: false,
        colorBlindSafe: false
      },
      ...initialSettings.theme
    },
    playback: {
      isPlaying: false,
      speed: 1.0,
      currentTime: new Date(),
      startTime: new Date(Date.now() - 3600000),
      endTime: new Date(),
      ...initialSettings.playback
    }
  });

  // Filter state
  const [filterGroups, setFilterGroups] = useState<FilterGroup[]>([]);
  const [filterPresets, setFilterPresets] = useState<FilterPreset[]>([]);
  const [activeFilterPreset, setActiveFilterPreset] = useState<string | undefined>();

  // Predefined themes
  const colorThemes: ColorTheme[] = [
    {
      name: 'dark',
      primary: '#3b82f6',
      secondary: '#6b7280',
      accent: '#10b981',
      background: '#1f2937',
      text: '#f9fafb',
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444'
    },
    {
      name: 'light',
      primary: '#2563eb',
      secondary: '#6b7280',
      accent: '#059669',
      background: '#ffffff',
      text: '#1f2937',
      success: '#059669',
      warning: '#d97706',
      error: '#dc2626'
    },
    {
      name: 'high-contrast',
      primary: '#000000',
      secondary: '#666666',
      accent: '#ffffff',
      background: '#ffffff',
      text: '#000000',
      success: '#000000',
      warning: '#000000',
      error: '#000000'
    }
  ];

  // Camera presets
  const cameraPresets: CameraPreset[] = [
    { name: 'Overview', position: [0, 100, 100], target: [0, 0, 0] },
    { name: 'Close-up', position: [20, 20, 20], target: [0, 0, 0] },
    { name: 'Side View', position: [100, 0, 0], target: [0, 0, 0] },
    { name: 'Top Down', position: [0, 150, 0], target: [0, 0, 0] },
    { name: 'Cognitive Focus', position: [30, 50, 80], target: [0, 10, 0] },
    { name: 'Memory Focus', position: [-50, 30, 60], target: [-10, 0, 0] }
  ];

  // Initialize filter system
  useEffect(() => {
    const presets = filteringSystem.getPresets();
    setFilterPresets(presets);
    setFilterGroups(filteringSystem.getState().groups);

    const listener = () => {
      setFilterGroups(filteringSystem.getState().groups);
    };

    filteringSystem.addListener('controls', listener);

    return () => {
      filteringSystem.removeListener('controls');
    };
  }, []);

  // Notify parent of settings changes
  useEffect(() => {
    onSettingsChange(settings);
  }, [settings, onSettingsChange]);

  // Layer controls
  const updateLayerVisibility = useCallback((layer: keyof LayerVisibility, visible: boolean) => {
    setSettings(prev => ({
      ...prev,
      layers: {
        ...prev.layers,
        [layer]: visible
      }
    }));
  }, []);

  const toggleAllLayers = useCallback((visible: boolean) => {
    setSettings(prev => ({
      ...prev,
      layers: Object.keys(prev.layers).reduce((acc, key) => ({
        ...acc,
        [key]: visible
      }), {} as LayerVisibility)
    }));
  }, []);

  // Quality controls
  const updateQuality = useCallback((key: keyof VisualizationSettings['quality'], value: any) => {
    setSettings(prev => ({
      ...prev,
      quality: {
        ...prev.quality,
        [key]: value
      }
    }));
  }, []);

  const setQualityPreset = useCallback((preset: 'low' | 'medium' | 'high' | 'ultra') => {
    const presets = {
      low: { particleCount: 1000, renderQuality: 'low', antiAliasing: false, shadows: false, postProcessing: false },
      medium: { particleCount: 2500, renderQuality: 'medium', antiAliasing: true, shadows: false, postProcessing: false },
      high: { particleCount: 5000, renderQuality: 'high', antiAliasing: true, shadows: true, postProcessing: true },
      ultra: { particleCount: 10000, renderQuality: 'ultra', antiAliasing: true, shadows: true, postProcessing: true }
    };

    setSettings(prev => ({
      ...prev,
      quality: {
        ...prev.quality,
        ...presets[preset],
        renderQuality: preset
      }
    }));
  }, []);

  // Camera controls
  const applyCameraPreset = useCallback((presetName: string) => {
    setSettings(prev => ({
      ...prev,
      camera: {
        ...prev.camera,
        currentPreset: presetName
      }
    }));
  }, []);

  // Theme controls
  const setTheme = useCallback((themeName: string) => {
    setSettings(prev => ({
      ...prev,
      theme: {
        ...prev.theme,
        current: themeName
      }
    }));
  }, []);

  const updateAccessibility = useCallback((key: keyof VisualizationSettings['theme']['accessibility'], value: boolean) => {
    setSettings(prev => ({
      ...prev,
      theme: {
        ...prev.theme,
        accessibility: {
          ...prev.theme.accessibility,
          [key]: value
        }
      }
    }));
  }, []);

  // Playback controls
  const updatePlayback = useCallback((updates: Partial<PlaybackState>) => {
    setSettings(prev => ({
      ...prev,
      playback: {
        ...prev.playback,
        ...updates
      }
    }));
  }, []);

  const togglePlayback = useCallback(() => {
    updatePlayback({ isPlaying: !settings.playback.isPlaying });
  }, [settings.playback.isPlaying, updatePlayback]);

  const resetTimeRange = useCallback(() => {
    const now = new Date();
    updatePlayback({
      currentTime: now,
      startTime: new Date(now.getTime() - 3600000), // 1 hour ago
      endTime: now,
      isPlaying: false
    });
  }, [updatePlayback]);

  // Filter controls
  const addFilterGroup = useCallback(() => {
    const name = prompt('Enter filter group name:');
    if (name) {
      filteringSystem.addFilterGroup(name);
    }
  }, []);

  const removeFilterGroup = useCallback((groupId: string) => {
    filteringSystem.removeFilterGroup(groupId);
  }, []);

  const loadFilterPreset = useCallback((presetId: string) => {
    if (filteringSystem.loadPreset(presetId)) {
      setActiveFilterPreset(presetId);
    }
  }, []);

  const saveCurrentAsPreset = useCallback(() => {
    const name = prompt('Enter preset name:');
    if (name) {
      const description = prompt('Enter description (optional):') || '';
      filteringSystem.savePreset(name, description, 'custom');
      setFilterPresets(filteringSystem.getPresets());
    }
  }, []);

  // Render control panel
  const renderLayerControls = () => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Layer Visibility</h3>
        <div className="space-x-2">
          <button
            onClick={() => toggleAllLayers(true)}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Show All
          </button>
          <button
            onClick={() => toggleAllLayers(false)}
            className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Hide All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {Object.entries(settings.layers).map(([layer, visible]) => (
          <label key={layer} className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visible}
              onChange={(e) => updateLayerVisibility(layer as keyof LayerVisibility, e.target.checked)}
              className="rounded"
            />
            <span className="capitalize">{layer.replace(/([A-Z])/g, ' $1').trim()}</span>
          </label>
        ))}
      </div>

      <div className="border-t pt-4">
        <h4 className="font-medium mb-3">Playback Controls</h4>
        <div className="flex items-center space-x-4 mb-4">
          <button
            onClick={togglePlayback}
            className={`px-4 py-2 rounded ${settings.playback.isPlaying 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-green-600 hover:bg-green-700'
            } text-white`}
          >
            {settings.playback.isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={resetTimeRange}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Reset
          </button>
        </div>

        <div className="space-y-2">
          <label className="block">
            <span className="text-sm">Playback Speed: {settings.playback.speed}x</span>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={settings.playback.speed}
              onChange={(e) => updatePlayback({ speed: parseFloat(e.target.value) })}
              className="w-full mt-1"
            />
          </label>
        </div>
      </div>
    </div>
  );

  const renderFilterControls = () => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Data Filtering</h3>
        <button
          onClick={addFilterGroup}
          className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Add Group
        </button>
      </div>

      {/* Filter presets */}
      <div>
        <h4 className="font-medium mb-2">Quick Presets</h4>
        <div className="grid grid-cols-2 gap-2">
          {filterPresets.slice(0, 6).map(preset => (
            <button
              key={preset.id}
              onClick={() => loadFilterPreset(preset.id)}
              className={`px-3 py-2 text-sm rounded border ${
                activeFilterPreset === preset.id
                  ? 'bg-blue-100 border-blue-500 text-blue-800'
                  : 'bg-gray-100 border-gray-300 hover:bg-gray-200'
              }`}
            >
              {preset.name}
            </button>
          ))}
        </div>
        <button
          onClick={saveCurrentAsPreset}
          className="mt-2 w-full px-3 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
        >
          Save Current as Preset
        </button>
      </div>

      {/* Active filter groups */}
      <div>
        <h4 className="font-medium mb-2">Active Filters</h4>
        {filterGroups.length === 0 ? (
          <p className="text-gray-500 text-sm">No active filters</p>
        ) : (
          <div className="space-y-2">
            {filterGroups.map(group => (
              <div key={group.id} className="p-3 border rounded bg-gray-50">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">{group.name}</span>
                  <button
                    onClick={() => removeFilterGroup(group.id)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  </button>
                </div>
                <p className="text-sm text-gray-600">
                  {group.conditions.length} condition(s), {group.operator} logic
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderQualityControls = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Quality Settings</h3>
      
      <div>
        <h4 className="font-medium mb-2">Quality Presets</h4>
        <div className="grid grid-cols-2 gap-2">
          {(['low', 'medium', 'high', 'ultra'] as const).map(preset => (
            <button
              key={preset}
              onClick={() => setQualityPreset(preset)}
              className={`px-3 py-2 text-sm rounded ${
                settings.quality.renderQuality === preset
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              {preset.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <label className="block">
          <span className="text-sm">Particle Count: {settings.quality.particleCount.toLocaleString()}</span>
          <input
            type="range"
            min="500"
            max="20000"
            step="500"
            value={settings.quality.particleCount}
            onChange={(e) => updateQuality('particleCount', parseInt(e.target.value))}
            className="w-full mt-1"
          />
        </label>

        <div className="space-y-2">
          {([
            { key: 'antiAliasing', label: 'Anti-aliasing' },
            { key: 'shadows', label: 'Shadows' },
            { key: 'postProcessing', label: 'Post Processing' }
          ] as const).map(({ key, label }) => (
            <label key={key} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.quality[key]}
                onChange={(e) => updateQuality(key, e.target.checked)}
                className="rounded"
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  const renderCameraControls = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Camera Controls</h3>
      
      <div>
        <h4 className="font-medium mb-2">Presets</h4>
        <div className="grid grid-cols-2 gap-2">
          {cameraPresets.map(preset => (
            <button
              key={preset.name}
              onClick={() => applyCameraPreset(preset.name)}
              className={`px-3 py-2 text-sm rounded ${
                settings.camera.currentPreset === preset.name
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        {([
          { key: 'smoothTransitions', label: 'Smooth Transitions' },
          { key: 'followMouse', label: 'Follow Mouse' },
          { key: 'autoRotate', label: 'Auto Rotate' }
        ] as const).map(({ key, label }) => (
          <label key={key} className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.camera[key]}
              onChange={(e) => setSettings(prev => ({
                ...prev,
                camera: { ...prev.camera, [key]: e.target.checked }
              }))}
              className="rounded"
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderThemeControls = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Theme & Accessibility</h3>
      
      <div>
        <h4 className="font-medium mb-2">Color Theme</h4>
        <div className="space-y-2">
          {colorThemes.map(theme => (
            <label key={theme.name} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="theme"
                checked={settings.theme.current === theme.name}
                onChange={() => setTheme(theme.name)}
                className="rounded"
              />
              <span className="capitalize">{theme.name.replace('-', ' ')}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h4 className="font-medium mb-2">Accessibility</h4>
        <div className="space-y-2">
          {([
            { key: 'highContrast', label: 'High Contrast' },
            { key: 'reducedMotion', label: 'Reduced Motion' },
            { key: 'colorBlindSafe', label: 'Color Blind Safe' }
          ] as const).map(({ key, label }) => (
            <label key={key} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.theme.accessibility[key]}
                onChange={(e) => updateAccessibility(key, e.target.checked)}
                className="rounded"
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`fixed top-4 left-4 bg-white rounded-lg shadow-lg border transition-all duration-300 ${
      isExpanded ? 'w-96 h-[80vh]' : 'w-64 h-auto'
    } z-50`}>
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center">
        <h2 className="text-xl font-bold text-gray-800">Visualization Controls</h2>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-gray-600 hover:text-gray-800"
        >
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {/* Quick actions */}
      {!isExpanded && (
        <div className="p-4 space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => onExportRequest('screenshot')}
              className="px-3 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
            >
              Screenshot
            </button>
            <button
              onClick={() => onDebugToggle(true)}
              className="px-3 py-2 text-sm bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              Debug
            </button>
          </div>
          <button
            onClick={() => onPerformanceMonitor(true)}
            className="w-full px-3 py-2 text-sm bg-yellow-600 text-white rounded hover:bg-yellow-700"
          >
            Performance Monitor
          </button>
        </div>
      )}

      {/* Expanded controls */}
      {isExpanded && (
        <div className="flex flex-col h-full">
          {/* Tabs */}
          <div className="flex border-b">
            {([
              { key: 'layers', label: 'Layers' },
              { key: 'filters', label: 'Filters' },
              { key: 'quality', label: 'Quality' },
              { key: 'camera', label: 'Camera' },
              { key: 'theme', label: 'Theme' }
            ] as const).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-4 py-2 text-sm border-b-2 ${
                  activeTab === key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-600 hover:text-gray-800'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-4">
            {activeTab === 'layers' && renderLayerControls()}
            {activeTab === 'filters' && renderFilterControls()}
            {activeTab === 'quality' && renderQualityControls()}
            {activeTab === 'camera' && renderCameraControls()}
            {activeTab === 'theme' && renderThemeControls()}
          </div>

          {/* Footer actions */}
          <div className="border-t p-4">
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => onExportRequest('screenshot')}
                className="px-3 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
              >
                Screenshot
              </button>
              <button
                onClick={() => onExportRequest('video')}
                className="px-3 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700"
              >
                Video
              </button>
              <button
                onClick={() => onExportRequest('data')}
                className="px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Data
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <button
                onClick={() => onDebugToggle(true)}
                className="px-3 py-2 text-sm bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                Debug Console
              </button>
              <button
                onClick={() => onPerformanceMonitor(true)}
                className="px-3 py-2 text-sm bg-yellow-600 text-white rounded hover:bg-yellow-700"
              >
                Performance
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualizationControls;