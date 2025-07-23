import React, { useState, useEffect, useMemo } from 'react';
import { Box, Grid, Paper, Typography, Tabs, Tab, Alert, CircularProgress, IconButton, Tooltip, Badge } from '@mui/material';
import { Dashboard, Psychology, Memory, Api, BugReport, Analytics, School, Search, Architecture, Refresh } from '@mui/icons-material';
import { useWebSocket } from '../../providers/WebSocketProvider';
import { useAppSelector } from '../../stores';
import { BrainKnowledgeGraph } from '../../components/visualizations/BrainKnowledgeGraph';
import { NeuralActivationHeatmap } from '../../components/visualizations/NeuralActivationHeatmap';
import { CognitiveSystemsDashboard } from '../../components/cognitive/CognitiveSystemsDashboard';
import { MemorySystemsMonitor } from '../../components/memory/MemorySystemsMonitor';
import { SystemMetricsOverview } from '../../components/monitoring/SystemMetricsOverview';
import { APIDataFlowMonitor } from '../../components/api/APIDataFlowMonitor';
import { ErrorDetectionDashboard } from '../../components/debugging/ErrorDetectionDashboard';
import { LearningAdaptationMonitor } from '../../components/learning/LearningAdaptationMonitor';
import { QuerySearchAnalytics } from '../../components/analytics/QuerySearchAnalytics';
import { ArchitectureDependencies } from '../../components/architecture/ArchitectureDependencies';
import { BrainGraphData, BrainMetrics } from '../../types/brain';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`llmkg-tabpanel-${index}`}
      aria-labelledby={`llmkg-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
    </div>
  );
};

const LLMKGDashboard: React.FC = () => {
  const { isConnected, connectionState } = useWebSocket();
  const currentData = useAppSelector(state => state.data.current);
  const [activeTab, setActiveTab] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);

  // Transform WebSocket data to component-specific formats
  const brainGraphData = useMemo<BrainGraphData | null>(() => {
    if (!currentData) return null;

    // Extract brain metrics from the WebSocket data
    const metrics = (currentData as any).metrics || {};
    
    // Transform entities from the knowledge graph nodes
    const entities = currentData.knowledgeGraph?.nodes?.map((node: any) => ({
      id: node.id,
      type_id: node.type === 'concept' ? 1 : node.type === 'entity' ? 2 : 3,
      properties: node.metadata || {},
      embedding: node.embedding || [],
      activation: node.weight || 0,
      direction: node.type === 'concept' ? 'Hidden' : 'Input',
      lastActivation: Date.now(),
      lastUpdate: Date.now(),
      conceptIds: []
    })) || [];

    // Transform relationships from edges
    const relationships = currentData.knowledgeGraph?.edges?.map((edge: any) => ({
      from: edge.source,
      to: edge.target,
      relType: 1,
      weight: edge.weight || 0.5,
      inhibitory: edge.type === 'inhibitory',
      temporalDecay: 0.1,
      lastActivation: Date.now(),
      usageCount: 1
    })) || [];

    // Calculate statistics
    const stats = {
      entityCount: entities.length,
      relationshipCount: relationships.length,
      avgActivation: entities.reduce((sum: number, e: any) => sum + e.activation, 0) / entities.length || 0,
      minActivation: Math.min(...entities.map((e: any) => e.activation)) || 0,
      maxActivation: Math.max(...entities.map((e: any) => e.activation)) || 1,
      totalActivation: entities.reduce((sum: number, e: any) => sum + e.activation, 0),
      graphDensity: currentData.knowledgeGraph?.metrics?.density || 0,
      clusteringCoefficient: metrics.brain_clustering_coefficient || 0,
      betweennessCentrality: 0,
      learningEfficiency: metrics.brain_learning_efficiency || 0,
      conceptCoherence: metrics.brain_concept_coherence || 0,
      activeEntities: metrics.brain_active_entities || entities.length,
      avgRelationshipsPerEntity: relationships.length / entities.length || 0,
      uniqueEntityTypes: new Set(entities.map((e: any) => e.type_id)).size
    };

    // Calculate activation distribution
    const activationDist = {
      veryLow: entities.filter((e: any) => e.activation < 0.2).length,
      low: entities.filter((e: any) => e.activation >= 0.2 && e.activation < 0.4).length,
      medium: entities.filter((e: any) => e.activation >= 0.4 && e.activation < 0.6).length,
      high: entities.filter((e: any) => e.activation >= 0.6 && e.activation < 0.8).length,
      veryHigh: entities.filter((e: any) => e.activation >= 0.8).length
    };

    return {
      entities,
      relationships,
      concepts: [],
      logicGates: [],
      statistics: stats,
      activationDistribution: activationDist,
      metrics: metrics as BrainMetrics
    };
  }, [currentData]);

  // Extract cognitive systems data
  const cognitiveData = useMemo(() => {
    if (!currentData) return null;

    return {
      patterns: [
        {
          id: 'convergent',
          name: 'Convergent Thinking',
          type: 'convergent' as const,
          strength: 0.7,
          confidence: 0.85,
          active: true,
          resources: 25,
          effectiveness: 0.8,
          lastSwitch: Date.now()
        },
        {
          id: 'divergent',
          name: 'Divergent Thinking',
          type: 'divergent' as const,
          strength: 0.6,
          confidence: 0.75,
          active: false,
          resources: 20,
          effectiveness: 0.7,
          lastSwitch: Date.now()
        },
        {
          id: 'lateral',
          name: 'Lateral Thinking',
          type: 'lateral' as const,
          strength: 0.5,
          confidence: 0.65,
          active: false,
          resources: 15,
          effectiveness: 0.6,
          lastSwitch: Date.now()
        },
        {
          id: 'systems',
          name: 'Systems Thinking',
          type: 'systems' as const,
          strength: 0.8,
          confidence: 0.9,
          active: true,
          resources: 30,
          effectiveness: 0.85,
          lastSwitch: Date.now()
        },
        {
          id: 'critical',
          name: 'Critical Thinking',
          type: 'critical' as const,
          strength: 0.75,
          confidence: 0.88,
          active: false,
          resources: 10,
          effectiveness: 0.82,
          lastSwitch: Date.now()
        }
      ],
      attention: {
        targets: [
          { id: '1', name: 'Pattern Recognition', priority: 0.8, resources: 30, type: 'pattern' as const },
          { id: '2', name: 'Memory Consolidation', priority: 0.6, resources: 25, type: 'task' as const },
          { id: '3', name: 'Entity Analysis', priority: 0.7, resources: 20, type: 'entity' as const },
          { id: '4', name: 'Concept Formation', priority: 0.5, resources: 25, type: 'concept' as const }
        ],
        totalCapacity: 100,
        usedCapacity: 100
      },
      inhibitory: {
        connections: [],
        balance: 0.1,
        patterns: ['noise_suppression', 'redundancy_filter']
      },
      patternHistory: [],
      executiveCommands: []
    };
  }, [currentData]);

  // Extract memory systems data
  const memoryData = useMemo(() => {
    if (!currentData) return null;

    return {
      workingMemory: {
        buffers: [
          {
            id: 'visual',
            name: 'Visual Buffer',
            capacity: 100,
            used: currentData.memory?.workingMemory?.usage || 45,
            items: [],
            decayRate: 0.5,
            accessPattern: 'spatial' as const
          },
          {
            id: 'verbal',
            name: 'Verbal Buffer',
            capacity: 100,
            used: 30,
            items: [],
            decayRate: 0.3,
            accessPattern: 'sequential' as const
          },
          {
            id: 'executive',
            name: 'Executive Buffer',
            capacity: 50,
            used: 20,
            items: [],
            decayRate: 0.1,
            accessPattern: 'random' as const
          }
        ],
        totalCapacity: 250,
        totalUsed: currentData.memory?.workingMemory?.usage || 95
      },
      longTermMemory: {
        consolidationRate: currentData.memory?.longTermMemory?.consolidationRate || 0.85,
        retrievalSpeed: currentData.memory?.longTermMemory?.retrievalSpeed || 50,
        totalItems: 1000,
        indexSize: 1024 * 1024,
        compressionRatio: 0.65
      },
      consolidation: {
        processes: [],
        queue: []
      },
      sdr: {
        patterns: [],
        totalBits: 2048,
        averageSparsity: 0.02
      },
      zeroCopy: {
        enabled: true,
        mappedRegions: 5,
        totalMappedSize: 1024 * 1024 * 10,
        accessLatency: 5
      },
      forgettingCurve: []
    };
  }, [currentData]);

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  const tabs = [
    { label: 'Overview', icon: <Dashboard /> },
    { label: 'Brain Graph', icon: <Psychology /> },
    { label: 'Neural Activity', icon: <Psychology /> },
    { label: 'Cognitive Systems', icon: <Psychology /> },
    { label: 'Memory', icon: <Memory /> },
    { label: 'Learning', icon: <School /> },
    { label: 'API Flow', icon: <Api /> },
    { label: 'Errors', icon: <BugReport /> },
    { label: 'Analytics', icon: <Analytics /> },
    { label: 'Architecture', icon: <Architecture /> }
  ];

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Connection Status */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          WebSocket disconnected. Attempting to reconnect...
        </Alert>
      )}

      {/* Tab Navigation */}
      <Paper sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', px: 2 }}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            aria-label="LLMKG dashboard tabs"
            sx={{ flexGrow: 1 }}
          >
            {tabs.map((tab, index) => (
              <Tab
                key={index}
                label={tab.label}
                icon={tab.icon}
                iconPosition="start"
                id={`llmkg-tab-${index}`}
                aria-controls={`llmkg-tabpanel-${index}`}
              />
            ))}
          </Tabs>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Paper>

      {/* Content */}
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {/* Overview Tab */}
        <TabPanel value={activeTab} index={0}>
          <SystemMetricsOverview data={currentData} />
        </TabPanel>

        {/* Brain Graph Tab */}
        <TabPanel value={activeTab} index={1}>
          {brainGraphData ? (
            <BrainKnowledgeGraph data={brainGraphData} height={700} />
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
              <CircularProgress />
            </Box>
          )}
        </TabPanel>

        {/* Neural Activity Tab */}
        <TabPanel value={activeTab} index={2}>
          {brainGraphData ? (
            <NeuralActivationHeatmap
              entities={brainGraphData.entities}
              activationDistribution={brainGraphData.activationDistribution}
              height={700}
            />
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
              <CircularProgress />
            </Box>
          )}
        </TabPanel>

        {/* Cognitive Systems Tab */}
        <TabPanel value={activeTab} index={3}>
          {cognitiveData ? (
            <CognitiveSystemsDashboard data={cognitiveData} height={700} />
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
              <CircularProgress />
            </Box>
          )}
        </TabPanel>

        {/* Memory Tab */}
        <TabPanel value={activeTab} index={4}>
          {memoryData ? (
            <MemorySystemsMonitor data={memoryData} height={700} />
          ) : (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
              <CircularProgress />
            </Box>
          )}
        </TabPanel>

        {/* Learning Tab */}
        <TabPanel value={activeTab} index={5}>
          <LearningAdaptationMonitor />
        </TabPanel>

        {/* API Flow Tab */}
        <TabPanel value={activeTab} index={6}>
          <APIDataFlowMonitor />
        </TabPanel>

        {/* Errors Tab */}
        <TabPanel value={activeTab} index={7}>
          <ErrorDetectionDashboard />
        </TabPanel>

        {/* Analytics Tab */}
        <TabPanel value={activeTab} index={8}>
          <QuerySearchAnalytics />
        </TabPanel>

        {/* Architecture Tab */}
        <TabPanel value={activeTab} index={9}>
          <ArchitectureDependencies />
        </TabPanel>
      </Box>
    </Box>
  );
};

export default LLMKGDashboard;