import React, { useState, useCallback } from 'react';
import { Layout, Layouts } from 'react-grid-layout';
import {
  GridLayout,
  LayoutManager,
  ResizablePanel,
  ResponsiveContainer,
  ViewportOptimizer,
  type LayoutItem,
  type LayoutPreset
} from '../components/Layout';
import { 
  NeuralActivityHeatmap,
  KnowledgeGraph3D,
  CognitivePatternViz,
  MemorySystemChart
} from '../components/visualizations';
import { MetricCard, ActivityFeed, DataGrid } from '../components/common';
import { useResponsiveLayout } from '../hooks/useResponsiveLayout';

interface DemoComponentProps {
  title: string;
  type: 'visualization' | 'metric' | 'activity' | 'data';
  color?: string;
}

const DemoComponent: React.FC<DemoComponentProps> = ({ title, type, color = '#007bff' }) => {
  const getIcon = () => {
    switch (type) {
      case 'visualization': return '📊';
      case 'metric': return '📈';
      case 'activity': return '⚡';
      case 'data': return '📋';
      default: return '📦';
    }
  };

  return (
    <div style={{
      width: '100%',
      height: '100%',
      padding: '16px',
      backgroundColor: '#f8f9fa',
      border: '1px solid #e9ecef',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
      boxSizing: 'border-box'
    }}>
      <div style={{ fontSize: '2rem', marginBottom: '12px' }}>
        {getIcon()}
      </div>
      <h3 style={{ 
        margin: '0 0 8px 0', 
        color: color,
        fontSize: '16px',
        fontWeight: 'bold'
      }}>
        {title}
      </h3>
      <p style={{ 
        margin: 0, 
        color: '#6c757d', 
        fontSize: '12px'
      }}>
        {type.charAt(0).toUpperCase() + type.slice(1)} Component
      </p>
      <div style={{
        marginTop: '12px',
        padding: '4px 8px',
        backgroundColor: color,
        color: 'white',
        borderRadius: '4px',
        fontSize: '10px',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        LLMKG
      </div>
    </div>
  );
};

const LayoutSystemDemo: React.FC = () => {
  const { currentBreakpoint, windowSize, isMobile, isTablet, isDesktop } = useResponsiveLayout();
  
  const [demoItems, setDemoItems] = useState<LayoutItem[]>([
    {
      i: 'neural-heatmap',
      x: 0,
      y: 0,
      w: 6,
      h: 4,
      minW: 3,
      minH: 3,
      component: (
        <DemoComponent 
          title="Neural Activity Heatmap" 
          type="visualization" 
          color="#dc3545" 
        />
      )
    },
    {
      i: 'knowledge-graph',
      x: 6,
      y: 0,
      w: 6,
      h: 4,
      minW: 4,
      minH: 3,
      component: (
        <DemoComponent 
          title="3D Knowledge Graph" 
          type="visualization" 
          color="#28a745" 
        />
      )
    },
    {
      i: 'cognitive-patterns',
      x: 0,
      y: 4,
      w: 4,
      h: 3,
      minW: 2,
      minH: 2,
      component: (
        <DemoComponent 
          title="Cognitive Patterns" 
          type="visualization" 
          color="#ffc107" 
        />
      )
    },
    {
      i: 'memory-metrics',
      x: 4,
      y: 4,
      w: 4,
      h: 3,
      minW: 2,
      minH: 2,
      component: (
        <DemoComponent 
          title="Memory System" 
          type="metric" 
          color="#17a2b8" 
        />
      )
    },
    {
      i: 'activity-feed',
      x: 8,
      y: 4,
      w: 4,
      h: 3,
      minW: 2,
      minH: 2,
      component: (
        <DemoComponent 
          title="Activity Feed" 
          type="activity" 
          color="#6f42c1" 
        />
      )
    },
    {
      i: 'performance-data',
      x: 0,
      y: 7,
      w: 6,
      h: 2,
      minW: 3,
      minH: 1,
      component: (
        <DemoComponent 
          title="Performance Data Grid" 
          type="data" 
          color="#fd7e14" 
        />
      )
    },
    {
      i: 'system-status',
      x: 6,
      y: 7,
      w: 6,
      h: 2,
      minW: 3,
      minH: 1,
      component: (
        <DemoComponent 
          title="System Status" 
          type="metric" 
          color="#20c997" 
        />
      )
    }
  ]);

  const handleLayoutChange = useCallback((layout: Layout[], layouts: Layouts) => {
    console.log('Layout changed:', { layout, layouts });
  }, []);

  const handleItemsChange = useCallback((items: LayoutItem[]) => {
    setDemoItems(items);
  }, []);

  const demoStats = {
    totalItems: demoItems.length,
    currentBreakpoint,
    windowSize: `${windowSize.width}x${windowSize.height}`,
    deviceType: isMobile ? 'Mobile' : isTablet ? 'Tablet' : isDesktop ? 'Desktop' : 'Unknown'
  };

  return (
    <div style={{ padding: '20px', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <div style={{ marginBottom: '30px' }}>
        <h1 style={{ 
          margin: '0 0 10px 0', 
          color: '#343a40',
          fontSize: '2.5rem',
          fontWeight: 'bold'
        }}>
          LLMKG Advanced Layout System
        </h1>
        <p style={{ 
          margin: '0 0 20px 0', 
          color: '#6c757d',
          fontSize: '1.1rem'
        }}>
          Responsive, drag-and-drop grid layout with viewport optimization for brain-inspired visualizations
        </p>

        {/* Stats Bar */}
        <div style={{
          display: 'flex',
          gap: '15px',
          flexWrap: 'wrap',
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          marginBottom: '20px'
        }}>
          {Object.entries(demoStats).map(([key, value]) => (
            <div key={key} style={{ textAlign: 'center', minWidth: '120px' }}>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#007bff' }}>
                {value}
              </div>
              <div style={{ fontSize: '0.875rem', color: '#6c757d', textTransform: 'capitalize' }}>
                {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Layout Demos */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '40px' }}>
        
        {/* 1. Basic Grid Layout Demo */}
        <section>
          <h2 style={{ marginBottom: '15px', color: '#495057' }}>
            1. Basic Grid Layout with Drag & Drop
          </h2>
          <ResponsiveContainer
            style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '20px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              minHeight: '600px'
            }}
          >
            <GridLayout
              items={demoItems}
              onLayoutChange={handleLayoutChange}
              isDraggable={true}
              isResizable={true}
              margin={[15, 15]}
              containerPadding={[0, 0]}
            />
          </ResponsiveContainer>
        </section>

        {/* 2. Layout Manager Demo */}
        <section>
          <h2 style={{ marginBottom: '15px', color: '#495057' }}>
            2. Layout Manager with Presets
          </h2>
          <ResponsiveContainer
            style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '20px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              minHeight: '600px'
            }}
          >
            <LayoutManager
              items={demoItems}
              onItemsChange={handleItemsChange}
              allowPresets={true}
              allowCustomization={true}
              category="cognitive"
            />
          </ResponsiveContainer>
        </section>

        {/* 3. Resizable Panel Demo */}
        <section>
          <h2 style={{ marginBottom: '15px', color: '#495057' }}>
            3. Resizable Panel Components
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '20px'
          }}>
            <ResizablePanel
              width={350}
              height={250}
              minWidth={200}
              minHeight={150}
              maxWidth={500}
              maxHeight={400}
              isResizable={true}
              style={{
                backgroundColor: 'white',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            >
              <DemoComponent 
                title="Resizable Neural Monitor" 
                type="visualization" 
                color="#e83e8c" 
              />
            </ResizablePanel>

            <ResizablePanel
              width={350}
              height={250}
              minWidth={200}
              minHeight={150}
              isResizable={true}
              aspectRatio={1.4}
              maintainAspectRatio={true}
              style={{
                backgroundColor: 'white',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            >
              <DemoComponent 
                title="Aspect Ratio Locked" 
                type="metric" 
                color="#6610f2" 
              />
            </ResizablePanel>
          </div>
        </section>

        {/* 4. Viewport Optimizer Demo */}
        <section>
          <h2 style={{ marginBottom: '15px', color: '#495057' }}>
            4. Viewport Optimization & Lazy Loading
          </h2>
          <ResponsiveContainer
            style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '20px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              height: '400px',
              overflow: 'hidden'
            }}
          >
            <ViewportOptimizer
              enableLazyLoading={true}
              enableVirtualization={true}
              itemHeight={180}
              maxVisibleItems={20}
              onVisibilityChange={(visibleIds) => {
                console.log('Visible items:', visibleIds);
              }}
              onPerformanceMetrics={(metrics) => {
                console.log('Performance:', metrics);
              }}
            >
              {Array.from({ length: 50 }, (_, index) => (
                <div key={index} style={{ marginBottom: '10px' }}>
                  <DemoComponent 
                    title={`Virtualized Item ${index + 1}`}
                    type={['visualization', 'metric', 'activity', 'data'][index % 4] as any}
                    color={[
                      '#007bff', '#28a745', '#dc3545', '#ffc107', 
                      '#17a2b8', '#6f42c1', '#e83e8c', '#fd7e14'
                    ][index % 8]}
                  />
                </div>
              ))}
            </ViewportOptimizer>
          </ResponsiveContainer>
        </section>

        {/* 5. Responsive Breakpoints Demo */}
        <section>
          <h2 style={{ marginBottom: '15px', color: '#495057' }}>
            5. Responsive Container Breakpoints
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '15px'
          }}>
            {['xs', 'sm', 'md', 'lg', 'xl'].map(breakpoint => (
              <ResponsiveContainer
                key={breakpoint}
                breakpoint={breakpoint}
                style={{
                  backgroundColor: 'white',
                  borderRadius: '8px',
                  padding: '20px',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                  minHeight: '150px',
                  textAlign: 'center'
                }}
              >
                <div style={{
                  fontSize: '1.25rem',
                  fontWeight: 'bold',
                  color: '#007bff',
                  marginBottom: '10px'
                }}>
                  {breakpoint.toUpperCase()}
                </div>
                <div style={{ color: '#6c757d', fontSize: '0.875rem' }}>
                  Breakpoint: {breakpoint}<br/>
                  Active: {currentBreakpoint === breakpoint ? '✅' : '❌'}
                </div>
              </ResponsiveContainer>
            ))}
          </div>
        </section>
      </div>

      {/* Footer with usage instructions */}
      <div style={{
        marginTop: '40px',
        padding: '20px',
        backgroundColor: '#343a40',
        color: 'white',
        borderRadius: '8px'
      }}>
        <h3 style={{ marginTop: 0, color: '#ffc107' }}>Usage Instructions</h3>
        <ul style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
          <li><strong>Drag & Drop:</strong> Click and drag items to rearrange the layout</li>
          <li><strong>Resize:</strong> Drag the resize handles (corners) to change panel sizes</li>
          <li><strong>Presets:</strong> Use the layout manager to save and load custom arrangements</li>
          <li><strong>Responsive:</strong> Resize your browser window to see adaptive layouts</li>
          <li><strong>Performance:</strong> Scroll the viewport optimizer section to see lazy loading</li>
        </ul>
        
        <div style={{ marginTop: '15px', fontSize: '0.875rem', color: '#adb5bd' }}>
          <strong>LLMKG Dashboard Layout System</strong> - Advanced responsive layouts for brain-inspired AI visualizations
        </div>
      </div>
    </div>
  );
};

export default LayoutSystemDemo;