import React, { useState } from 'react';
import { useAppDispatch, useAppSelector, dashboardActions } from '../stores';
import { setLayoutSettings } from '../stores/slices/layoutSlice';

const SettingsPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const config = useAppSelector(state => state.dashboard.config);
  const layoutSettings = useAppSelector(state => state.layout.settings);
  const [primaryColor, setPrimaryColor] = useState('#3B82F6');

  const handleThemeChange = (theme: 'light' | 'dark' | 'auto') => {
    dispatch(dashboardActions.setTheme(theme));
  };

  const handleRefreshRateChange = (rate: number) => {
    dispatch(dashboardActions.setRefreshRate(rate));
  };

  const handleMaxDataPointsChange = (points: number) => {
    dispatch(dashboardActions.setMaxDataPoints(points));
  };

  const handleAnimationsToggle = () => {
    dispatch(dashboardActions.toggleAnimations());
  };

  const handleLayoutToggle = () => {
    dispatch(setLayoutSettings({ 
      compactType: layoutSettings.compactType === 'vertical' ? 'horizontal' : 'vertical' 
    }));
  };

  const handleApplyTheme = () => {
    // Update CSS variable for primary color
    document.documentElement.style.setProperty('--color-primary', primaryColor);
  };

  return (
    <div className="settings-page">
      <h1>Dashboard Settings</h1>
      <p>Configure dashboard preferences and performance settings.</p>
      
      <div className="settings-sections">
        <section className="settings-section">
          <h2>Appearance</h2>
          
          <div className="setting-item">
            <label>Theme</label>
            <div className="theme-buttons">
              {(['light', 'dark', 'auto'] as const).map(theme => (
                <button
                  key={theme}
                  className={`theme-button ${config.theme === theme ? 'active' : ''}`}
                  onClick={() => handleThemeChange(theme)}
                >
                  {theme.charAt(0).toUpperCase() + theme.slice(1)}
                </button>
              ))}
            </div>
          </div>
          
          <div className="setting-item">
            <label htmlFor="primary-color">Primary Color</label>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <input
                id="primary-color"
                type="color"
                value={primaryColor}
                onChange={(e) => setPrimaryColor(e.target.value)}
                aria-label="Primary color"
              />
              <input
                type="text"
                value={primaryColor}
                onChange={(e) => setPrimaryColor(e.target.value)}
                style={{ width: '100px' }}
              />
              <button
                onClick={handleApplyTheme}
                className="apply-button"
              >
                Apply
              </button>
            </div>
          </div>
          
          <div className="setting-item">
            <label>
              <input
                type="checkbox"
                checked={config.enableAnimations}
                onChange={handleAnimationsToggle}
              />
              Enable Animations
            </label>
          </div>
        </section>
        
        <section className="settings-section">
          <h2>Layout</h2>
          
          <div className="setting-item">
            <button
              onClick={handleLayoutToggle}
              className="layout-toggle-button"
              aria-label="Toggle layout"
            >
              Toggle Layout Direction
            </button>
            <p className="setting-description">
              Current: {layoutSettings.compactType === 'vertical' ? 'Vertical' : 'Horizontal'} compaction
            </p>
          </div>
        </section>
        
        <section className="settings-section">
          <h2>Performance</h2>
          
          <div className="setting-item">
            <label>Refresh Rate (ms)</label>
            <select
              value={config.refreshRate}
              onChange={e => handleRefreshRateChange(Number(e.target.value))}
            >
              <option value={500}>500ms (High)</option>
              <option value={1000}>1000ms (Normal)</option>
              <option value={2000}>2000ms (Low)</option>
              <option value={5000}>5000ms (Very Low)</option>
            </select>
          </div>
          
          <div className="setting-item">
            <label>Max Data Points</label>
            <select
              value={config.maxDataPoints}
              onChange={e => handleMaxDataPointsChange(Number(e.target.value))}
            >
              <option value={500}>500</option>
              <option value={1000}>1000</option>
              <option value={2000}>2000</option>
              <option value={5000}>5000</option>
            </select>
          </div>
        </section>
        
        <section className="settings-section">
          <h2>Current Configuration</h2>
          <pre className="config-display">
            {JSON.stringify(config, null, 2)}
          </pre>
        </section>
      </div>

      <style jsx>{`
        .settings-page {
          padding: 2rem;
          max-width: 800px;
        }
        
        .settings-sections {
          margin-top: 2rem;
        }
        
        .settings-section {
          margin-bottom: 3rem;
        }
        
        .settings-section h2 {
          color: var(--text-primary, #ffffff);
          margin-bottom: 1.5rem;
          font-size: 1.25rem;
          border-bottom: 1px solid var(--border-color, #404040);
          padding-bottom: 0.5rem;
        }
        
        .setting-item {
          margin-bottom: 1.5rem;
        }
        
        .setting-item label {
          display: block;
          color: var(--text-primary, #ffffff);
          margin-bottom: 0.5rem;
          font-weight: 500;
        }
        
        .setting-item input[type="checkbox"] {
          margin-right: 0.5rem;
        }
        
        .setting-item select {
          background: var(--bg-secondary, #2d2d2d);
          color: var(--text-primary, #ffffff);
          border: 1px solid var(--border-color, #404040);
          border-radius: 4px;
          padding: 0.5rem;
          font-size: 0.875rem;
        }
        
        .theme-buttons {
          display: flex;
          gap: 0.5rem;
        }
        
        .theme-button {
          background: var(--bg-secondary, #2d2d2d);
          color: var(--text-primary, #ffffff);
          border: 1px solid var(--border-color, #404040);
          border-radius: 4px;
          padding: 0.5rem 1rem;
          cursor: pointer;
          font-size: 0.875rem;
          transition: all 0.2s;
        }
        
        .theme-button:hover {
          background: var(--bg-tertiary, #404040);
        }
        
        .theme-button.active {
          background: var(--accent-color, #007acc);
          color: white;
          border-color: var(--accent-color, #007acc);
        }
        
        .config-display {
          background: var(--bg-secondary, #2d2d2d);
          border: 1px solid var(--border-color, #404040);
          border-radius: 4px;
          padding: 1rem;
          font-size: 0.875rem;
          color: var(--text-secondary, #b3b3b3);
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
};

export default SettingsPage;