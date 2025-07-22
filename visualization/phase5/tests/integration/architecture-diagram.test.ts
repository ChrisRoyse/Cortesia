import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { SystemArchitectureDiagram } from '../../src/components/SystemArchitectureDiagram';
import { ArchitectureDiagramEngine } from '../../src/core/ArchitectureDiagramEngine';
import { LayoutEngine } from '../../src/core/LayoutEngine';
import { 
  createMockArchitectureData, 
  mockWebSocketConnection, 
  MockTelemetryProvider 
} from '../utils/test-helpers';

describe('Architecture Diagram Integration Tests', () => {
  let store: any;
  let mockData: any;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        architecture: (state = {}, action: any) => state
      }
    });
    mockData = createMockArchitectureData();
  });

  describe('Diagram Rendering and Layout', () => {
    test('renders architecture diagram with all components', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            layout="neural-layers"
            realTimeEnabled={true}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
        expect(screen.getByTestId('svg-container')).toBeInTheDocument();
      });

      // Verify all nodes are rendered
      mockData.nodes.forEach((node: any) => {
        expect(screen.getByTestId(`node-${node.id}`)).toBeInTheDocument();
      });

      // Verify all connections are rendered
      mockData.connections.forEach((connection: any) => {
        expect(screen.getByTestId(`connection-${connection.id}`)).toBeInTheDocument();
      });
    });

    test('applies neural-layers layout correctly', async () => {
      const layoutEngine = new LayoutEngine();
      const layoutResult = layoutEngine.applyLayout(mockData, 'neural-layers');

      expect(layoutResult.nodes).toBeDefined();
      expect(layoutResult.layers).toBeDefined();

      // Verify hierarchical positioning
      const corticalNodes = layoutResult.nodes.filter((n: any) => 
        n.layer === 'cognitive-cortical'
      );
      const subcorticalNodes = layoutResult.nodes.filter((n: any) => 
        n.layer === 'cognitive-subcortical'
      );

      // Cortical nodes should be positioned above subcortical
      corticalNodes.forEach((cortical: any) => {
        subcorticalNodes.forEach((subcortical: any) => {
          expect(cortical.position.y).toBeLessThan(subcortical.position.y);
        });
      });
    });

    test('handles different layout algorithms', async () => {
      const layouts = ['neural-layers', 'hierarchical', 'force-directed', 'circular', 'grid'];
      const layoutEngine = new LayoutEngine();

      for (const layout of layouts) {
        const result = layoutEngine.applyLayout(mockData, layout);
        
        expect(result.nodes).toBeDefined();
        expect(result.nodes.length).toBe(mockData.nodes.length);
        
        // Each layout should produce different positioning
        result.nodes.forEach((node: any) => {
          expect(node.position).toBeDefined();
          expect(typeof node.position.x).toBe('number');
          expect(typeof node.position.y).toBe('number');
        });
      }
    });
  });

  describe('Node Interactions', () => {
    test('handles node selection and multi-selection', async () => {
      const onNodeClick = jest.fn();
      const onSelectionChange = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            onNodeClick={onNodeClick}
            onSelectionChange={onSelectionChange}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      const firstNode = screen.getByTestId(`node-${mockData.nodes[0].id}`);
      const secondNode = screen.getByTestId(`node-${mockData.nodes[1].id}`);

      // Single selection
      fireEvent.click(firstNode);
      expect(onNodeClick).toHaveBeenCalledWith(mockData.nodes[0]);
      expect(onSelectionChange).toHaveBeenCalledWith([mockData.nodes[0]]);

      // Multi-selection with Ctrl+click
      fireEvent.click(secondNode, { ctrlKey: true });
      expect(onSelectionChange).toHaveBeenCalledWith([
        mockData.nodes[0],
        mockData.nodes[1]
      ]);
    });

    test('handles node drag and drop', async () => {
      const onNodePositionChange = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            onNodePositionChange={onNodePositionChange}
            enableDragAndDrop={true}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      const node = screen.getByTestId(`node-${mockData.nodes[0].id}`);

      // Simulate drag
      fireEvent.mouseDown(node, { clientX: 100, clientY: 100 });
      fireEvent.mouseMove(node, { clientX: 150, clientY: 150 });
      fireEvent.mouseUp(node);

      await waitFor(() => {
        expect(onNodePositionChange).toHaveBeenCalled();
      });
    });

    test('displays node details on hover', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            showTooltips={true}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      const node = screen.getByTestId(`node-${mockData.nodes[0].id}`);

      fireEvent.mouseEnter(node);
      
      await waitFor(() => {
        expect(screen.getByTestId('node-tooltip')).toBeInTheDocument();
        expect(screen.getByText(mockData.nodes[0].label)).toBeInTheDocument();
        expect(screen.getByText(/CPU:/)).toBeInTheDocument();
        expect(screen.getByText(/Memory:/)).toBeInTheDocument();
      });
    });
  });

  describe('Connection Visualization', () => {
    test('renders connections with proper styling', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            showConnectionFlow={true}
          />
        </Provider>
      );

      await waitFor(() => {
        mockData.connections.forEach((connection: any) => {
          const connectionElement = screen.getByTestId(`connection-${connection.id}`);
          expect(connectionElement).toBeInTheDocument();
          
          // Check connection styling based on type
          if (connection.type === 'excitation') {
            expect(connectionElement).toHaveClass('connection-excitation');
          } else if (connection.type === 'inhibition') {
            expect(connectionElement).toHaveClass('connection-inhibition');
          }
        });
      });
    });

    test('animates data flow on active connections', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            showConnectionFlow={true}
            enableAnimations={true}
          />
        </Provider>
      );

      await waitFor(() => {
        const activeConnections = mockData.connections.filter((c: any) => c.active);
        activeConnections.forEach((connection: any) => {
          const flowElement = screen.getByTestId(`flow-${connection.id}`);
          expect(flowElement).toBeInTheDocument();
          expect(flowElement).toHaveClass('flow-animation');
        });
      });
    });

    test('handles connection click events', async () => {
      const onConnectionClick = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            onConnectionClick={onConnectionClick}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      const connection = screen.getByTestId(`connection-${mockData.connections[0].id}`);
      fireEvent.click(connection);

      expect(onConnectionClick).toHaveBeenCalledWith(mockData.connections[0]);
    });
  });

  describe('Layer Visualization', () => {
    test('renders cognitive layers with proper hierarchy', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            showLayers={true}
          />
        </Provider>
      );

      await waitFor(() => {
        mockData.layers.forEach((layer: any) => {
          const layerElement = screen.getByTestId(`layer-${layer.id}`);
          expect(layerElement).toBeInTheDocument();
          
          // Check layer ordering
          const layerOrder = parseInt(layerElement.getAttribute('data-order') || '0');
          expect(layerOrder).toBe(layer.order);
        });
      });
    });

    test('groups nodes by layer correctly', async () => {
      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            showLayers={true}
          />
        </Provider>
      );

      await waitFor(() => {
        mockData.layers.forEach((layer: any) => {
          const layerContainer = screen.getByTestId(`layer-container-${layer.id}`);
          
          layer.nodes.forEach((nodeId: string) => {
            const node = screen.getByTestId(`node-${nodeId}`);
            expect(layerContainer).toContainElement(node);
          });
        });
      });
    });
  });

  describe('Performance Optimization', () => {
    test('handles large datasets efficiently', async () => {
      const largeData = createMockArchitectureData(150); // 150 nodes
      const startTime = performance.now();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={largeData}
            enableVirtualization={true}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      const renderTime = performance.now() - startTime;
      expect(renderTime).toBeLessThan(1000); // Should render in less than 1 second
    });

    test('uses virtualization for large datasets', async () => {
      const largeData = createMockArchitectureData(200);

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={largeData}
            enableVirtualization={true}
            viewportSize={{ width: 800, height: 600 }}
          />
        </Provider>
      );

      await waitFor(() => {
        const visibleNodes = screen.getAllByTestId(/node-/).length;
        // Should render fewer nodes than total when virtualization is active
        expect(visibleNodes).toBeLessThan(largeData.nodes.length);
      });
    });
  });

  describe('Keyboard Navigation', () => {
    test('supports keyboard shortcuts', async () => {
      const onZoomIn = jest.fn();
      const onZoomOut = jest.fn();
      const onReset = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            onZoomIn={onZoomIn}
            onZoomOut={onZoomOut}
            onReset={onReset}
            enableKeyboardNavigation={true}
          />
        </Provider>
      );

      const diagram = screen.getByTestId('architecture-diagram');
      diagram.focus();

      // Test zoom shortcuts
      fireEvent.keyDown(diagram, { key: '+', ctrlKey: true });
      expect(onZoomIn).toHaveBeenCalled();

      fireEvent.keyDown(diagram, { key: '-', ctrlKey: true });
      expect(onZoomOut).toHaveBeenCalled();

      fireEvent.keyDown(diagram, { key: '0', ctrlKey: true });
      expect(onReset).toHaveBeenCalled();
    });

    test('supports arrow key navigation', async () => {
      const onSelectionChange = jest.fn();

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            onSelectionChange={onSelectionChange}
            enableKeyboardNavigation={true}
          />
        </Provider>
      );

      const diagram = screen.getByTestId('architecture-diagram');
      diagram.focus();

      // Select first node with Enter
      fireEvent.keyDown(diagram, { key: 'Enter' });
      
      // Navigate with arrow keys
      fireEvent.keyDown(diagram, { key: 'ArrowRight' });
      
      await waitFor(() => {
        expect(onSelectionChange).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles malformed data gracefully', async () => {
      const malformedData = {
        nodes: [
          { id: 'test-1' }, // Missing required fields
          null,
          undefined
        ],
        connections: [
          { sourceId: 'nonexistent', targetId: 'also-nonexistent' }
        ]
      };

      const consoleError = jest.spyOn(console, 'error').mockImplementation(() => {});

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={malformedData as any}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('architecture-diagram')).toBeInTheDocument();
      });

      // Should not crash the application
      expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
      
      consoleError.mockRestore();
    });

    test('displays error messages for invalid connections', async () => {
      const dataWithInvalidConnections = {
        ...mockData,
        connections: [
          {
            id: 'invalid-conn',
            sourceId: 'nonexistent-source',
            targetId: 'nonexistent-target',
            type: 'excitation'
          }
        ]
      };

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={dataWithInvalidConnections}
            showValidationErrors={true}
          />
        </Provider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('validation-errors')).toBeInTheDocument();
        expect(screen.getByText(/Invalid connection/)).toBeInTheDocument();
      });
    });
  });

  describe('Theme Integration', () => {
    test('applies custom theme correctly', async () => {
      const customTheme = {
        colors: {
          primary: '#ff6b35',
          cognitive: {
            cortical: '#4ecdc4',
            subcortical: '#45b7d1'
          }
        }
      };

      render(
        <Provider store={store}>
          <SystemArchitectureDiagram
            architectureData={mockData}
            theme={customTheme}
          />
        </Provider>
      );

      await waitFor(() => {
        const diagramRoot = screen.getByTestId('architecture-diagram');
        const computedStyle = getComputedStyle(diagramRoot);
        
        expect(computedStyle.getPropertyValue('--primary-color')).toBe('#ff6b35');
        expect(computedStyle.getPropertyValue('--cortical-color')).toBe('#4ecdc4');
      });
    });
  });
});