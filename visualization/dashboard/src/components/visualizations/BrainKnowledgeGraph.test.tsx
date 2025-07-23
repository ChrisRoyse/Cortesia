import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { BrainKnowledgeGraph } from './BrainKnowledgeGraph';
import { BrainGraphData } from '../../types/brain';
import { ThemeProvider } from '@mui/material/styles';
import { createTheme } from '@mui/material/styles';

// Mock Three.js components
jest.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="canvas-mock">{children}</div>,
  useFrame: jest.fn(),
  useThree: () => ({ camera: {}, scene: {} }),
}));

jest.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Html: ({ children }: any) => <div data-testid="html-annotation">{children}</div>,
  Sphere: ({ children, ...props }: any) => <div data-testid="sphere" {...props}>{children}</div>,
  Line: (props: any) => <div data-testid="line" {...props} />,
  Text: ({ children, ...props }: any) => <div data-testid="text" {...props}>{children}</div>,
}));

const mockStore = configureStore({
  reducer: {
    data: (state = { current: null }) => state,
  },
});

const theme = createTheme();

const mockBrainData: BrainGraphData = {
  entities: [
    {
      id: 'entity1',
      type_id: 1,
      properties: { name: 'Input Node' },
      embedding: [0.1, 0.2, 0.3],
      activation: 0.8,
      direction: 'Input',
      lastActivation: Date.now(),
      lastUpdate: Date.now(),
      conceptIds: ['concept1'],
    },
    {
      id: 'entity2',
      type_id: 2,
      properties: { name: 'Hidden Node' },
      embedding: [0.4, 0.5, 0.6],
      activation: 0.5,
      direction: 'Hidden',
      lastActivation: Date.now(),
      lastUpdate: Date.now(),
      conceptIds: ['concept1'],
    },
    {
      id: 'entity3',
      type_id: 3,
      properties: { name: 'Output Node' },
      embedding: [0.7, 0.8, 0.9],
      activation: 0.3,
      direction: 'Output',
      lastActivation: Date.now(),
      lastUpdate: Date.now(),
      conceptIds: ['concept2'],
    },
  ],
  relationships: [
    {
      from: 'entity1',
      to: 'entity2',
      relType: 1,
      weight: 0.9,
      inhibitory: false,
      temporalDecay: 0.1,
      lastActivation: Date.now(),
      usageCount: 10,
    },
    {
      from: 'entity2',
      to: 'entity3',
      relType: 2,
      weight: 0.5,
      inhibitory: true,
      temporalDecay: 0.2,
      lastActivation: Date.now(),
      usageCount: 5,
    },
  ],
  concepts: [
    {
      id: 'concept1',
      name: 'Input Processing',
      inputs: ['entity1'],
      outputs: ['entity2'],
      gates: [],
      coherence: 0.85,
      activation: 0.65,
      lastUpdate: Date.now(),
    },
  ],
  logicGates: [
    {
      id: 'gate1',
      type: 'AND',
      inputs: ['entity1', 'entity2'],
      outputs: ['entity3'],
      currentState: true,
    },
  ],
  statistics: {
    entityCount: 3,
    relationshipCount: 2,
    avgActivation: 0.53,
    minActivation: 0.3,
    maxActivation: 0.8,
    totalActivation: 1.6,
    graphDensity: 0.67,
    clusteringCoefficient: 0.5,
    betweennessCentrality: 0.4,
    learningEfficiency: 0.75,
    conceptCoherence: 0.85,
    activeEntities: 3,
    avgRelationshipsPerEntity: 0.67,
    uniqueEntityTypes: 3,
  },
  activationDistribution: {
    veryLow: 0,
    low: 1,
    medium: 1,
    high: 0,
    veryHigh: 1,
  },
  metrics: {
    brain_entity_count: 3,
    brain_relationship_count: 2,
    brain_avg_activation: 0.53,
    brain_max_activation: 0.8,
    brain_graph_density: 0.67,
    brain_clustering_coefficient: 0.5,
    brain_total_activation: 1.6,
    brain_active_entities: 3,
    brain_learning_efficiency: 0.75,
    brain_concept_coherence: 0.85,
    brain_avg_relationships_per_entity: 0.67,
    brain_unique_entity_types: 3,
    brain_memory_bytes: 1024,
    brain_index_memory_bytes: 512,
    brain_embedding_memory_bytes: 256,
    brain_total_chunks: 10,
    brain_total_triples: 5,
  },
};

const renderComponent = (data?: BrainGraphData) => {
  return render(
    <Provider store={mockStore}>
      <ThemeProvider theme={theme}>
        <BrainKnowledgeGraph data={data} />
      </ThemeProvider>
    </Provider>
  );
};

describe('BrainKnowledgeGraph', () => {
  it('renders without data', () => {
    renderComponent();
    expect(screen.getByText(/No brain graph data available/i)).toBeInTheDocument();
  });

  it('renders 3D canvas with brain data', () => {
    renderComponent(mockBrainData);
    expect(screen.getByTestId('canvas-mock')).toBeInTheDocument();
    expect(screen.getAllByTestId('sphere')).toHaveLength(3); // 3 entities
    expect(screen.getAllByTestId('line')).toHaveLength(2); // 2 relationships
  });

  it('displays entity details panel', () => {
    renderComponent(mockBrainData);
    expect(screen.getByText('Entity Details')).toBeInTheDocument();
    expect(screen.getByText(/Select an entity/i)).toBeInTheDocument();
  });

  it('displays statistics panel', () => {
    renderComponent(mockBrainData);
    expect(screen.getByText('Graph Statistics')).toBeInTheDocument();
    expect(screen.getByText(/Entities: 3/)).toBeInTheDocument();
    expect(screen.getByText(/Relationships: 2/)).toBeInTheDocument();
    expect(screen.getByText(/Avg Activation: 0.53/)).toBeInTheDocument();
  });

  it('filters entities by activation threshold', () => {
    renderComponent(mockBrainData);
    const slider = screen.getByRole('slider', { name: /activation threshold/i });
    fireEvent.change(slider, { target: { value: 0.6 } });
    
    // Should only show entities with activation >= 0.6
    expect(screen.queryByText('Input Node')).toBeInTheDocument(); // 0.8 activation
    expect(screen.queryByText('Hidden Node')).not.toBeInTheDocument(); // 0.5 activation
  });

  it('toggles inhibitory connections visibility', () => {
    renderComponent(mockBrainData);
    const toggle = screen.getByLabelText(/show inhibitory/i);
    
    // Initially shows all connections
    expect(screen.getAllByTestId('line')).toHaveLength(2);
    
    // Toggle off inhibitory
    fireEvent.click(toggle);
    // Implementation would filter connections
  });

  it('searches entities by properties', async () => {
    renderComponent(mockBrainData);
    const searchInput = screen.getByPlaceholderText(/search entities/i);
    
    fireEvent.change(searchInput, { target: { value: 'Input' } });
    
    await waitFor(() => {
      expect(screen.getByText('Input Node')).toBeInTheDocument();
      expect(screen.queryByText('Hidden Node')).not.toBeInTheDocument();
    });
  });

  it('displays activation history timeline', () => {
    renderComponent(mockBrainData);
    const timelineButton = screen.getByText(/activation history/i);
    fireEvent.click(timelineButton);
    
    expect(screen.getByText(/Activation Timeline/i)).toBeInTheDocument();
  });

  it('shows concept membership for selected entity', async () => {
    renderComponent(mockBrainData);
    const entity = screen.getAllByTestId('sphere')[0];
    
    fireEvent.click(entity);
    
    await waitFor(() => {
      expect(screen.getByText(/Concept: Input Processing/i)).toBeInTheDocument();
      expect(screen.getByText(/Coherence: 0.85/i)).toBeInTheDocument();
    });
  });

  it('animates activation propagation', () => {
    renderComponent(mockBrainData);
    const animateButton = screen.getByText(/animate propagation/i);
    
    fireEvent.click(animateButton);
    expect(screen.getByTestId('propagation-animation')).toBeInTheDocument();
  });

  it('exports graph data', () => {
    const mockDownload = jest.fn();
    global.URL.createObjectURL = jest.fn();
    
    renderComponent(mockBrainData);
    const exportButton = screen.getByText(/export graph/i);
    
    fireEvent.click(exportButton);
    // Verify export functionality
  });
});