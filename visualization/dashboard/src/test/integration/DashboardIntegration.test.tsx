import React from 'react';
import { screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { act } from 'react-dom/test-utils';
import App from '../../App';
import {
  renderWithProviders,
  MockWebSocket,
  setViewport,
  viewportSizes,
  waitForWebSocketConnection,
  measureRenderTime,
  mockServerMessages,
  expectNoConsoleErrors,
} from '../../utils/testUtils';
import { store } from '../../stores';

// Mock WebSocket globally
global.WebSocket = MockWebSocket as any;

describe('LLMKG Dashboard Phase 2 Integration Tests', () => {
  expectNoConsoleErrors();

  beforeEach(() => {
    // Reset store state
    store.dispatch({ type: 'RESET' });
    // Reset viewport
    setViewport(viewportSizes.desktop.width, viewportSizes.desktop.height);
    // Reset WebSocket instances
    MockWebSocket.instances = [];
    (global as any).WebSocket.instances = [];
  });

  describe('Component Integration', () => {
    test('all major components render without errors', async () => {
      renderWithProviders(<App />);
      
      // Wait for app to initialize
      await waitFor(() => {
        expect(screen.getByText(/LLMKG Dashboard/i)).toBeInTheDocument();
      });

      // Verify key components are present
      expect(screen.getByRole('navigation')).toBeInTheDocument();
      expect(screen.getByRole('main')).toBeInTheDocument();
    });

    test('theme system applies to all components', async () => {
      const { container } = renderWithProviders(<App />);
      
      // Check initial theme
      expect(container.firstChild).toHaveClass('light');
      
      // Find and click theme toggle
      const themeToggle = await screen.findByLabelText(/toggle theme/i);
      await userEvent.click(themeToggle);
      
      // Verify theme changed
      await waitFor(() => {
        expect(container.firstChild).toHaveClass('dark');
      });
    });

    test('responsive layouts work at different breakpoints', async () => {
      const { rerender } = renderWithProviders(<App />);
      
      // Test desktop
      setViewport(viewportSizes.desktop.width, viewportSizes.desktop.height);
      rerender(<App />);
      await waitFor(() => {
        const sidebar = screen.getByRole('complementary');
        expect(sidebar).toBeVisible();
      });
      
      // Test tablet
      setViewport(viewportSizes.tablet.width, viewportSizes.tablet.height);
      rerender(<App />);
      await waitFor(() => {
        // Sidebar should be collapsible on tablet
        const menuButton = screen.getByLabelText(/toggle menu/i);
        expect(menuButton).toBeInTheDocument();
      });
      
      // Test mobile
      setViewport(viewportSizes.mobile.width, viewportSizes.mobile.height);
      rerender(<App />);
      await waitFor(() => {
        // Sidebar should be hidden by default on mobile
        const sidebar = screen.queryByRole('complementary');
        expect(sidebar).not.toBeVisible();
      });
    });
  });

  describe('Data Flow Integration', () => {
    test('WebSocket data flows to visualization components', async () => {
      const { container } = renderWithProviders(<App />);
      
      // Navigate to Knowledge Graph page
      const knowledgeGraphLink = await screen.findByText(/Knowledge Graph/i);
      await userEvent.click(knowledgeGraphLink);
      
      // Wait for WebSocket connection
      const wsStatus = screen.getByTestId('ws-status');
      await waitFor(() => {
        expect(wsStatus).toHaveTextContent(/connected/i);
      });
      
      // Simulate server sending knowledge graph data
      // Find the WebSocket instance created by MockWebSocket
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      act(() => {
        if (ws.onmessage) {
          ws.onmessage(new MessageEvent('message', {
            data: JSON.stringify(mockServerMessages.knowledgeGraph)
          }));
        }
      });
      
      // Verify data is rendered
      await waitFor(() => {
        expect(screen.getByText(/Entity A/i)).toBeInTheDocument();
        expect(screen.getByText(/Entity B/i)).toBeInTheDocument();
      });
    });

    test('real-time updates work across multiple components', async () => {
      renderWithProviders(<App />);
      
      // Navigate to dashboard
      const dashboardLink = await screen.findByText(/Dashboard/i);
      await userEvent.click(dashboardLink);
      
      // Wait for components to load
      await waitFor(() => {
        expect(screen.getByText(/Cognitive Patterns/i)).toBeInTheDocument();
        expect(screen.getByText(/Memory Metrics/i)).toBeInTheDocument();
      });
      
      // Simulate multiple data updates
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      act(() => {
        ws.onmessage(new MessageEvent('message', {
          data: JSON.stringify(mockServerMessages.cognitivePatterns)
        }));
      });
      
      act(() => {
        ws.onmessage(new MessageEvent('message', {
          data: JSON.stringify(mockServerMessages.memoryMetrics)
        }));
      });
      
      // Verify both components updated
      await waitFor(() => {
        expect(screen.getByText(/Pattern Alpha/i)).toBeInTheDocument();
        expect(screen.getByText(/Working Memory/i)).toBeInTheDocument();
      });
    });

    test('state management handles concurrent updates', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      // Send rapid concurrent updates
      const updates = [];
      for (let i = 0; i < 10; i++) {
        updates.push(
          act(() => {
            ws.onmessage(new MessageEvent('message', {
              data: JSON.stringify({
                type: 'memory_metrics_update',
                data: {
                  ...mockServerMessages.memoryMetrics.data,
                  workingMemory: { ...mockServerMessages.memoryMetrics.data.workingMemory, used: i }
                }
              })
            }));
          })
        );
      }
      
      await Promise.all(updates);
      
      // Verify final state is consistent
      const state = store.getState();
      expect(state.realtime.memoryMetrics).toBeDefined();
    });
  });

  describe('Navigation Flow', () => {
    test('all routes render correctly', async () => {
      renderWithProviders(<App />);
      
      const routes = [
        { link: /Dashboard/i, content: /System Overview/i },
        { link: /Knowledge Graph/i, content: /Knowledge Graph Visualization/i },
        { link: /Neural Activity/i, content: /Neural Activity Monitor/i },
        { link: /Cognitive Patterns/i, content: /Cognitive Pattern Analysis/i },
        { link: /Memory Systems/i, content: /Memory System Status/i },
        { link: /Architecture/i, content: /System Architecture/i },
        { link: /Tools/i, content: /Development Tools/i },
        { link: /Settings/i, content: /Dashboard Settings/i },
      ];
      
      for (const route of routes) {
        const link = await screen.findByText(route.link);
        await userEvent.click(link);
        
        await waitFor(() => {
          expect(screen.getByText(route.content)).toBeInTheDocument();
        });
      }
    });

    test('breadcrumb navigation works correctly', async () => {
      renderWithProviders(<App />);
      
      // Navigate deep into the app
      const knowledgeGraphLink = await screen.findByText(/Knowledge Graph/i);
      await userEvent.click(knowledgeGraphLink);
      
      // Check breadcrumb
      const breadcrumb = screen.getByRole('navigation', { name: /breadcrumb/i });
      expect(within(breadcrumb).getByText(/Home/i)).toBeInTheDocument();
      expect(within(breadcrumb).getByText(/Knowledge Graph/i)).toBeInTheDocument();
      
      // Click home in breadcrumb
      const homeLink = within(breadcrumb).getByText(/Home/i);
      await userEvent.click(homeLink);
      
      // Verify navigation
      await waitFor(() => {
        expect(screen.getByText(/System Overview/i)).toBeInTheDocument();
      });
    });

    test('browser back/forward navigation works', async () => {
      const { container } = renderWithProviders(<App />);
      
      // Navigate to multiple pages
      const dashboardLink = await screen.findByText(/Dashboard/i);
      await userEvent.click(dashboardLink);
      
      const knowledgeGraphLink = await screen.findByText(/Knowledge Graph/i);
      await userEvent.click(knowledgeGraphLink);
      
      // Go back
      act(() => {
        window.history.back();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/System Overview/i)).toBeInTheDocument();
      });
      
      // Go forward
      act(() => {
        window.history.forward();
      });
      
      await waitFor(() => {
        expect(screen.getByText(/Knowledge Graph Visualization/i)).toBeInTheDocument();
      });
    });
  });

  describe('Layout System', () => {
    test('drag and drop functionality works', async () => {
      renderWithProviders(<App />);
      
      // Navigate to dashboard
      const dashboardLink = await screen.findByText(/Dashboard/i);
      await userEvent.click(dashboardLink);
      
      // Find draggable panels
      const panels = await screen.findAllByTestId(/draggable-panel/i);
      expect(panels.length).toBeGreaterThan(0);
      
      // Simulate drag and drop
      const firstPanel = panels[0];
      const secondPanel = panels[1];
      
      fireEvent.dragStart(firstPanel);
      fireEvent.dragOver(secondPanel);
      fireEvent.drop(secondPanel);
      fireEvent.dragEnd(firstPanel);
      
      // Verify layout updated
      const updatedPanels = await screen.findAllByTestId(/draggable-panel/i);
      expect(updatedPanels[0]).not.toBe(firstPanel);
    });

    test('resizable panels work correctly', async () => {
      renderWithProviders(<App />);
      
      // Find resizable panels
      const resizeHandles = await screen.findAllByTestId(/resize-handle/i);
      
      if (resizeHandles.length > 0) {
        const handle = resizeHandles[0];
        const initialPos = handle.getBoundingClientRect();
        
        // Simulate resize
        fireEvent.mouseDown(handle, { clientX: initialPos.left, clientY: initialPos.top });
        fireEvent.mouseMove(handle, { clientX: initialPos.left + 100, clientY: initialPos.top });
        fireEvent.mouseUp(handle);
        
        // Verify panel resized
        const newPos = handle.getBoundingClientRect();
        expect(newPos.left).not.toBe(initialPos.left);
      }
    });

    test('layout persistence works across sessions', async () => {
      const { unmount } = renderWithProviders(<App />);
      
      // Navigate to dashboard
      const dashboardLink = await screen.findByText(/Dashboard/i);
      await userEvent.click(dashboardLink);
      
      // Make layout changes
      const layoutToggle = await screen.findByLabelText(/toggle layout/i);
      await userEvent.click(layoutToggle);
      
      // Unmount and remount
      unmount();
      renderWithProviders(<App />);
      
      // Navigate back to dashboard
      const newDashboardLink = await screen.findByText(/Dashboard/i);
      await userEvent.click(newDashboardLink);
      
      // Verify layout persisted
      const state = store.getState();
      expect(state.layout).toBeDefined();
    });
  });

  describe('Theme Consistency', () => {
    test('theme changes apply to all components immediately', async () => {
      const { container } = renderWithProviders(<App />);
      
      // Get all themed elements
      const themedElements = container.querySelectorAll('[data-theme]');
      const initialThemes = Array.from(themedElements).map(el => 
        el.getAttribute('data-theme')
      );
      
      // Toggle theme
      const themeToggle = await screen.findByLabelText(/toggle theme/i);
      await userEvent.click(themeToggle);
      
      // Verify all elements updated
      await waitFor(() => {
        const updatedElements = container.querySelectorAll('[data-theme]');
        const updatedThemes = Array.from(updatedElements).map(el => 
          el.getAttribute('data-theme')
        );
        
        expect(updatedThemes).not.toEqual(initialThemes);
      });
    });

    test('custom theme colors apply correctly', async () => {
      renderWithProviders(<App />);
      
      // Navigate to settings
      const settingsLink = await screen.findByText(/Settings/i);
      await userEvent.click(settingsLink);
      
      // Find theme customization
      const primaryColorInput = await screen.findByLabelText(/primary color/i);
      await userEvent.clear(primaryColorInput);
      await userEvent.type(primaryColorInput, '#FF5733');
      
      // Apply changes
      const applyButton = screen.getByText(/Apply/i);
      await userEvent.click(applyButton);
      
      // Verify color applied
      const { container } = renderWithProviders(<App />);
      const primaryElements = container.querySelectorAll('.text-primary');
      
      primaryElements.forEach(el => {
        const styles = window.getComputedStyle(el);
        expect(styles.color).toContain('255, 87, 51'); // RGB values for #FF5733
      });
    });
  });

  describe('Performance', () => {
    test('dashboard renders within acceptable time', async () => {
      const renderTime = await measureRenderTime(<App />);
      
      expect(renderTime.mean).toBeLessThan(100); // 100ms average
      expect(renderTime.max).toBeLessThan(200); // 200ms worst case
    });

    test('real-time updates do not cause performance degradation', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      const startTime = performance.now();
      
      // Send 100 rapid updates
      for (let i = 0; i < 100; i++) {
        act(() => {
          ws.onmessage(new MessageEvent('message', {
            data: JSON.stringify(mockServerMessages.neuralActivity)
          }));
        });
      }
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      
      expect(totalTime).toBeLessThan(1000); // Should handle 100 updates in under 1s
    });

    test('memory usage remains stable during extended use', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      // Simulate extended use
      for (let i = 0; i < 50; i++) {
        // Navigate between pages
        const links = await screen.findAllByRole('link');
        const randomLink = links[Math.floor(Math.random() * links.length)];
        await userEvent.click(randomLink);
        
        // Send data updates
        act(() => {
          ws.onmessage(new MessageEvent('message', {
            data: JSON.stringify(mockServerMessages.knowledgeGraph)
          }));
        });
        
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be reasonable (less than 50MB)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });
  });

  describe('Phase 1 Integration', () => {
    test('connects to Phase 1 WebSocket server', async () => {
      renderWithProviders(<App />, { wsUrl: 'ws://localhost:8081' });
      
      // Check connection status
      const wsStatus = await screen.findByTestId('ws-status');
      
      await waitFor(() => {
        expect(wsStatus).toHaveTextContent(/connected/i);
      }, { timeout: 5000 });
    });

    test('handles Phase 1 server messages correctly', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      // Simulate Phase 1 server message format
      const phase1Message = {
        type: 'system_update',
        timestamp: Date.now(),
        data: {
          knowledge_graph: mockServerMessages.knowledgeGraph.data,
          cognitive_patterns: mockServerMessages.cognitivePatterns.data,
          memory_metrics: mockServerMessages.memoryMetrics.data,
          neural_activity: mockServerMessages.neuralActivity.data,
        }
      };
      
      act(() => {
        ws.onmessage(new MessageEvent('message', {
          data: JSON.stringify(phase1Message)
        }));
      });
      
      // Verify all data types are processed
      await waitFor(() => {
        const state = store.getState();
        expect(state.realtime.knowledgeGraph.current).toBeDefined();
        expect(state.realtime.cognitive.current).toBeDefined();
        expect(state.realtime.memory.current).toBeDefined();
        expect(state.realtime.neural.current).toBeDefined();
      });
    });

    test('gracefully handles connection failures', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      // Simulate connection error
      act(() => {
        ws.onerror(new Event('error'));
        ws.close();
      });
      
      // Check error handling
      const wsStatus = await screen.findByTestId('ws-status');
      await waitFor(() => {
        expect(wsStatus).toHaveTextContent(/disconnected|error/i);
      });
      
      // Verify reconnection attempt
      await waitFor(() => {
        expect(wsStatus).toHaveTextContent(/reconnecting/i);
      }, { timeout: 10000 });
    });
  });

  describe('Error Handling', () => {
    test('handles component errors gracefully', async () => {
      // Mock console.error to prevent test output pollution
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      // Create a component that throws
      const ThrowingComponent = () => {
        throw new Error('Test error');
      };
      
      renderWithProviders(
        <div>
          <ThrowingComponent />
        </div>
      );
      
      // Verify error boundary caught it
      await waitFor(() => {
        expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      });
      
      consoleSpy.mockRestore();
    });

    test('handles malformed WebSocket messages', async () => {
      renderWithProviders(<App />);
      
      const wsInstances = (global as any).WebSocket.instances || [];
      const ws = wsInstances[wsInstances.length - 1] || new MockWebSocket('ws://test');
      
      // Send malformed message
      act(() => {
        ws.onmessage(new MessageEvent('message', {
          data: 'invalid json {]'
        }));
      });
      
      // App should continue functioning
      await waitFor(() => {
        expect(screen.getByText(/LLMKG Dashboard/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    test('keyboard navigation works throughout the app', async () => {
      renderWithProviders(<App />);
      
      // Tab through interactive elements
      const user = userEvent.setup();
      
      // Start at the beginning
      await user.tab();
      
      // Verify focus moves through navigation
      const activeElement = document.activeElement;
      expect(activeElement?.tagName).toMatch(/A|BUTTON/i);
      
      // Tab through several elements
      for (let i = 0; i < 10; i++) {
        await user.tab();
      }
      
      // Verify we can activate elements with keyboard
      if (document.activeElement?.tagName === 'A') {
        await user.keyboard('{Enter}');
        
        // Verify navigation occurred
        await waitFor(() => {
          expect(window.location.pathname).not.toBe('/');
        });
      }
    });

    test('ARIA labels are present and correct', async () => {
      const { container } = renderWithProviders(<App />);
      
      // Check for required ARIA landmarks
      expect(screen.getByRole('navigation')).toBeInTheDocument();
      expect(screen.getByRole('main')).toBeInTheDocument();
      
      // Check interactive elements have labels
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });
      
      // Check form controls have labels
      const inputs = container.querySelectorAll('input, select, textarea');
      inputs.forEach(input => {
        const label = container.querySelector(`label[for="${input.id}"]`);
        expect(label || input.getAttribute('aria-label')).toBeTruthy();
      });
    });
  });
});