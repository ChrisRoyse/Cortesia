import React, { useState, useCallback } from 'react';
import { useSpring, animated } from '@react-spring/web';
import {
  NavigationControlsProps,
  LayoutType,
  ViewMode,
  ExportFormat
} from '../types';

const NavigationControls: React.FC<NavigationControlsProps> = ({
  layout,
  viewMode,
  selectedNodes,
  canUndo,
  canRedo,
  canExport,
  onLayoutChange,
  onViewModeChange,
  onZoomIn,
  onZoomOut,
  onZoomToFit,
  onResetView,
  onUndo,
  onRedo,
  onExport,
  onToggleHelp,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showLayoutMenu, setShowLayoutMenu] = useState(false);
  const [showViewModeMenu, setShowViewModeMenu] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);

  // Animation for control panel
  const panelSpring = useSpring({
    width: isExpanded ? 320 : 60,
    opacity: 1,
    config: { tension: 300, friction: 30 }
  });

  // Animation for expanded content
  const contentSpring = useSpring({
    opacity: isExpanded ? 1 : 0,
    transform: isExpanded ? 'translateX(0px)' : 'translateX(20px)',
    config: { tension: 300, friction: 30 }
  });

  const handleExport = useCallback((format: ExportFormat) => {
    onExport(format);
    setShowExportMenu(false);
  }, [onExport]);

  const handleLayoutChange = useCallback((newLayout: LayoutType) => {
    onLayoutChange(newLayout);
    setShowLayoutMenu(false);
  }, [onLayoutChange]);

  const handleViewModeChange = useCallback((newViewMode: ViewMode) => {
    onViewModeChange(newViewMode);
    setShowViewModeMenu(false);
  }, [onViewModeChange]);

  return (
    <animated.div
      className={`bg-white rounded-lg shadow-lg border overflow-hidden ${className}`}
      style={panelSpring}
    >
      {/* Main control bar */}
      <div className="flex items-center h-12 px-3">
        {/* Toggle button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-center w-8 h-8 rounded text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          title={isExpanded ? 'Collapse controls' : 'Expand controls'}
        >
          <span className={`transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
            ⚙️
          </span>
        </button>

        {/* Expanded controls */}
        <animated.div 
          style={contentSpring}
          className="flex items-center space-x-2 ml-3"
        >
          {isExpanded && (
            <>
              {/* Zoom controls */}
              <div className="flex items-center space-x-1 border-r pr-3">
                <ControlButton
                  icon="🔍+"
                  onClick={onZoomIn}
                  tooltip="Zoom In"
                  size="sm"
                />
                <ControlButton
                  icon="🔍-"
                  onClick={onZoomOut}
                  tooltip="Zoom Out"
                  size="sm"
                />
                <ControlButton
                  icon="⊞"
                  onClick={onZoomToFit}
                  tooltip="Zoom to Fit"
                  size="sm"
                />
                <ControlButton
                  icon="🏠"
                  onClick={onResetView}
                  tooltip="Reset View"
                  size="sm"
                />
              </div>

              {/* Layout controls */}
              <div className="relative border-r pr-3">
                <ControlButton
                  icon="📐"
                  onClick={() => setShowLayoutMenu(!showLayoutMenu)}
                  tooltip="Change Layout"
                  active={showLayoutMenu}
                />
                {showLayoutMenu && (
                  <LayoutMenu
                    currentLayout={layout}
                    onLayoutChange={handleLayoutChange}
                    onClose={() => setShowLayoutMenu(false)}
                  />
                )}
              </div>

              {/* View mode controls */}
              <div className="relative border-r pr-3">
                <ControlButton
                  icon="👁️"
                  onClick={() => setShowViewModeMenu(!showViewModeMenu)}
                  tooltip="Change View Mode"
                  active={showViewModeMenu}
                />
                {showViewModeMenu && (
                  <ViewModeMenu
                    currentViewMode={viewMode}
                    onViewModeChange={handleViewModeChange}
                    onClose={() => setShowViewModeMenu(false)}
                  />
                )}
              </div>

              {/* History controls */}
              <div className="flex items-center space-x-1 border-r pr-3">
                <ControlButton
                  icon="↶"
                  onClick={onUndo}
                  tooltip="Undo"
                  disabled={!canUndo}
                  size="sm"
                />
                <ControlButton
                  icon="↷"
                  onClick={onRedo}
                  tooltip="Redo"
                  disabled={!canRedo}
                  size="sm"
                />
              </div>

              {/* Export controls */}
              <div className="relative border-r pr-3">
                <ControlButton
                  icon="💾"
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  tooltip="Export Diagram"
                  disabled={!canExport}
                  active={showExportMenu}
                />
                {showExportMenu && (
                  <ExportMenu
                    onExport={handleExport}
                    onClose={() => setShowExportMenu(false)}
                  />
                )}
              </div>

              {/* Help button */}
              <ControlButton
                icon="❓"
                onClick={onToggleHelp}
                tooltip="Show Help"
              />
            </>
          )}
        </animated.div>
      </div>

      {/* Selection info */}
      {isExpanded && selectedNodes.length > 0 && (
        <animated.div style={contentSpring} className="border-t bg-gray-50 px-4 py-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">
              {selectedNodes.length} component{selectedNodes.length !== 1 ? 's' : ''} selected
            </span>
            <button
              onClick={() => {/* Clear selection */}}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              Clear
            </button>
          </div>
        </animated.div>
      )}

      {/* Current settings indicator */}
      {isExpanded && (
        <animated.div style={contentSpring} className="border-t bg-gray-50 px-4 py-2">
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
            <div>
              <span className="font-medium">Layout:</span> {getLayoutLabel(layout)}
            </div>
            <div>
              <span className="font-medium">View:</span> {getViewModeLabel(viewMode)}
            </div>
          </div>
        </animated.div>
      )}
    </animated.div>
  );
};

// Control button component
const ControlButton: React.FC<{
  icon: string;
  onClick: () => void;
  tooltip?: string;
  disabled?: boolean;
  active?: boolean;
  size?: 'sm' | 'md';
}> = ({ 
  icon, 
  onClick, 
  tooltip, 
  disabled = false, 
  active = false,
  size = 'md'
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={tooltip}
      className={`
        ${sizeClasses[size]}
        flex items-center justify-center rounded
        transition-colors duration-150
        ${disabled 
          ? 'text-gray-300 cursor-not-allowed' 
          : active
            ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }
      `}
    >
      {icon}
    </button>
  );
};

// Layout selection menu
const LayoutMenu: React.FC<{
  currentLayout: LayoutType;
  onLayoutChange: (layout: LayoutType) => void;
  onClose: () => void;
}> = ({ currentLayout, onLayoutChange, onClose }) => {
  const layouts: { id: LayoutType; label: string; description: string; icon: string }[] = [
    {
      id: 'neural-layers',
      label: 'Neural Layers',
      description: 'Brain-inspired hierarchical layout',
      icon: '🧠'
    },
    {
      id: 'hierarchical',
      label: 'Hierarchical',
      description: 'Top-down tree structure',
      icon: '📊'
    },
    {
      id: 'force-directed',
      label: 'Force-Directed',
      description: 'Physics-based automatic layout',
      icon: '🌐'
    },
    {
      id: 'circular',
      label: 'Circular',
      description: 'Circular arrangement',
      icon: '⭕'
    },
    {
      id: 'grid',
      label: 'Grid',
      description: 'Regular grid pattern',
      icon: '⊞'
    }
  ];

  return (
    <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-lg shadow-xl border z-10">
      <div className="p-3">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Layout Options</h3>
        <div className="space-y-1">
          {layouts.map(layout => (
            <button
              key={layout.id}
              onClick={() => onLayoutChange(layout.id)}
              className={`
                w-full text-left px-3 py-2 rounded-md text-sm transition-colors
                ${currentLayout === layout.id
                  ? 'bg-blue-100 text-blue-900'
                  : 'hover:bg-gray-100 text-gray-700'
                }
              `}
            >
              <div className="flex items-center space-x-3">
                <span className="text-lg">{layout.icon}</span>
                <div>
                  <div className="font-medium">{layout.label}</div>
                  <div className="text-xs text-gray-500">{layout.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="border-t p-2">
        <button
          onClick={onClose}
          className="w-full text-center text-xs text-gray-500 hover:text-gray-700"
        >
          Close
        </button>
      </div>
    </div>
  );
};

// View mode selection menu
const ViewModeMenu: React.FC<{
  currentViewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  onClose: () => void;
}> = ({ currentViewMode, onViewModeChange, onClose }) => {
  const viewModes: { id: ViewMode; label: string; description: string; icon: string }[] = [
    {
      id: 'overview',
      label: 'Overview',
      description: 'Complete system view',
      icon: '🌅'
    },
    {
      id: 'detailed',
      label: 'Detailed',
      description: 'Detailed component view',
      icon: '🔍'
    },
    {
      id: 'cognitive-layers',
      label: 'Cognitive Layers',
      description: 'Focus on brain layers',
      icon: '🧠'
    },
    {
      id: 'connections-focus',
      label: 'Connections',
      description: 'Emphasize connections',
      icon: '🔗'
    }
  ];

  return (
    <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-lg shadow-xl border z-10">
      <div className="p-3">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">View Modes</h3>
        <div className="space-y-1">
          {viewModes.map(mode => (
            <button
              key={mode.id}
              onClick={() => onViewModeChange(mode.id)}
              className={`
                w-full text-left px-3 py-2 rounded-md text-sm transition-colors
                ${currentViewMode === mode.id
                  ? 'bg-blue-100 text-blue-900'
                  : 'hover:bg-gray-100 text-gray-700'
                }
              `}
            >
              <div className="flex items-center space-x-3">
                <span className="text-lg">{mode.icon}</span>
                <div>
                  <div className="font-medium">{mode.label}</div>
                  <div className="text-xs text-gray-500">{mode.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="border-t p-2">
        <button
          onClick={onClose}
          className="w-full text-center text-xs text-gray-500 hover:text-gray-700"
        >
          Close
        </button>
      </div>
    </div>
  );
};

// Export options menu
const ExportMenu: React.FC<{
  onExport: (format: ExportFormat) => void;
  onClose: () => void;
}> = ({ onExport, onClose }) => {
  const exportOptions: { format: ExportFormat; label: string; description: string; icon: string }[] = [
    {
      format: 'svg',
      label: 'SVG Vector',
      description: 'Scalable vector graphics',
      icon: '📐'
    },
    {
      format: 'png',
      label: 'PNG Image',
      description: 'High-quality raster image',
      icon: '🖼️'
    },
    {
      format: 'json',
      label: 'JSON Data',
      description: 'Raw diagram data',
      icon: '📋'
    }
  ];

  return (
    <div className="absolute top-full right-0 mt-2 w-56 bg-white rounded-lg shadow-xl border z-10">
      <div className="p-3">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Export Options</h3>
        <div className="space-y-1">
          {exportOptions.map(option => (
            <button
              key={option.format}
              onClick={() => onExport(option.format)}
              className="w-full text-left px-3 py-2 rounded-md text-sm hover:bg-gray-100 text-gray-700 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <span className="text-lg">{option.icon}</span>
                <div>
                  <div className="font-medium">{option.label}</div>
                  <div className="text-xs text-gray-500">{option.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
      <div className="border-t p-2">
        <button
          onClick={onClose}
          className="w-full text-center text-xs text-gray-500 hover:text-gray-700"
        >
          Close
        </button>
      </div>
    </div>
  );
};

// Keyboard shortcuts display
const KeyboardShortcuts: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const shortcuts = [
    { key: 'Space', action: 'Pan mode' },
    { key: 'Ctrl + F', action: 'Zoom to fit' },
    { key: 'Ctrl + E', action: 'Export diagram' },
    { key: 'Ctrl + Z', action: 'Undo' },
    { key: 'Ctrl + Y', action: 'Redo' },
    { key: 'Escape', action: 'Clear selection' },
    { key: 'Delete', action: 'Hide selected' },
    { key: 'Arrow Keys', action: 'Navigate nodes' },
    { key: 'Enter', action: 'Activate focused node' },
    { key: '+/-', action: 'Zoom in/out' },
  ];

  return (
    <div className="absolute top-full right-0 mt-2 w-64 bg-white rounded-lg shadow-xl border z-20 max-h-80 overflow-y-auto">
      <div className="p-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-900">Keyboard Shortcuts</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            ×
          </button>
        </div>
        <div className="space-y-2">
          {shortcuts.map((shortcut, index) => (
            <div key={index} className="flex items-center justify-between text-xs">
              <span className="bg-gray-100 px-2 py-1 rounded font-mono">{shortcut.key}</span>
              <span className="text-gray-600">{shortcut.action}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Utility functions
function getLayoutLabel(layout: LayoutType): string {
  const labels = {
    'neural-layers': 'Neural Layers',
    'hierarchical': 'Hierarchical',
    'force-directed': 'Force-Directed',
    'circular': 'Circular',
    'grid': 'Grid'
  };
  return labels[layout] || layout;
}

function getViewModeLabel(viewMode: ViewMode): string {
  const labels = {
    'overview': 'Overview',
    'detailed': 'Detailed',
    'cognitive-layers': 'Cognitive',
    'connections-focus': 'Connections'
  };
  return labels[viewMode] || viewMode;
}

export default NavigationControls;