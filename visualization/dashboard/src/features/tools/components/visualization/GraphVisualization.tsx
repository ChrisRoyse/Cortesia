import React, { useEffect, useRef, useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  FormControl,
  Select,
  MenuItem,
  Slider,
  Switch,
  FormControlLabel,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  PhotoCamera,
  Tune,
  Timeline,
} from '@mui/icons-material';
import * as d3 from 'd3';
import { saveAs } from 'file-saver';

interface GraphNode {
  id: string;
  type: string;
  label?: string;
  properties?: Record<string, any>;
  x?: number;
  y?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight?: number;
  properties?: Record<string, any>;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata?: {
    title?: string;
    description?: string;
    nodeTypes?: string[];
    edgeTypes?: string[];
  };
}

interface GraphVisualizationProps {
  data: GraphData;
  fullscreen?: boolean;
  onNodeClick?: (node: GraphNode) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  data,
  fullscreen = false,
  onNodeClick,
  onEdgeClick,
}) => {
  const theme = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [layoutType, setLayoutType] = useState<'force' | 'hierarchical' | 'circular'>('force');
  const [showLabels, setShowLabels] = useState(true);
  const [nodeSize, setNodeSize] = useState(10);
  const [linkStrength, setLinkStrength] = useState(1);
  const [showControls, setShowControls] = useState(false);

  // Color schemes for different node and edge types
  const nodeColorScale = d3.scaleOrdinal(d3.schemeCategory10);
  const edgeColorScale = d3.scaleOrdinal(d3.schemePastel1);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data.nodes.length) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = fullscreen ? window.innerHeight - 200 : 600;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom as any);

    // Create main group for zoom/pan
    const g = svg.append('g');

    // Create arrow markers for directed edges
    svg.append('defs').selectAll('marker')
      .data(['default', 'selected'])
      .enter().append('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'selected' ? theme.palette.primary.main : theme.palette.text.secondary);

    // Prepare data
    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.edges.map(d => ({
      ...d,
      source: typeof d.source === 'string' ? d.source : d.source.id,
      target: typeof d.target === 'string' ? d.target : d.target.id,
    }));

    // Create force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links).id((d: any) => d.id).strength(linkStrength))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(nodeSize * 1.5));

    // Apply layout
    if (layoutType === 'hierarchical') {
      const stratify = d3.stratify()
        .id((d: any) => d.id)
        .parentId((d: any) => {
          const parentEdge = links.find(l => l.target === d.id);
          return parentEdge ? parentEdge.source : null;
        });

      try {
        const root = stratify(nodes);
        const treeLayout = d3.tree().size([width - 100, height - 100]);
        treeLayout(root as any);

        nodes.forEach((node: any) => {
          const treeNode = root.descendants().find(d => d.id === node.id);
          if (treeNode) {
            node.x = treeNode.x + 50;
            node.y = treeNode.y + 50;
            node.fx = node.x;
            node.fy = node.y;
          }
        });
      } catch (e) {
        // Fall back to force layout if hierarchical fails
        console.warn('Hierarchical layout failed, using force layout');
      }
    } else if (layoutType === 'circular') {
      const angleStep = (2 * Math.PI) / nodes.length;
      const radius = Math.min(width, height) / 3;
      nodes.forEach((node: any, i) => {
        node.x = width / 2 + radius * Math.cos(i * angleStep);
        node.y = height / 2 + radius * Math.sin(i * angleStep);
        node.fx = node.x;
        node.fy = node.y;
      });
    }

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', (d: any) => edgeColorScale(d.type))
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => Math.sqrt(d.weight || 1) * 2)
      .attr('marker-end', 'url(#arrow-default)')
      .on('click', (event, d) => {
        setSelectedEdge(d as any);
        onEdgeClick?.(d as any);
      })
      .on('mouseover', function() {
        d3.select(this)
          .attr('stroke-opacity', 1)
          .attr('stroke-width', (d: any) => Math.sqrt(d.weight || 1) * 3);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke-opacity', 0.6)
          .attr('stroke-width', (d: any) => Math.sqrt(d.weight || 1) * 2);
      });

    // Create nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
      .attr('r', nodeSize)
      .attr('fill', (d: any) => nodeColorScale(d.type))
      .attr('stroke', theme.palette.background.paper)
      .attr('stroke-width', 2)
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any)
      .on('click', (event, d) => {
        setSelectedNode(d as any);
        onNodeClick?.(d as any);
      })
      .on('mouseover', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', nodeSize * 1.5);
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', nodeSize);
      });

    // Add labels
    const label = g.append('g')
      .selectAll('text')
      .data(nodes)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -nodeSize - 5)
      .attr('font-size', '12px')
      .attr('fill', theme.palette.text.primary)
      .attr('opacity', showLabels ? 1 : 0)
      .text((d: any) => d.label || d.id);

    // Add tooltips
    node.append('title')
      .text((d: any) => `${d.type}: ${d.label || d.id}\n${JSON.stringify(d.properties || {}, null, 2)}`);

    link.append('title')
      .text((d: any) => `${d.type}: ${d.source.id || d.source} → ${d.target.id || d.target}\nWeight: ${d.weight || 1}`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Drag functions
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      if (layoutType === 'force') {
        d.fx = null;
        d.fy = null;
      }
    }

    // Export functions
    const exportSVG = () => {
      const svgData = new XMLSerializer().serializeToString(svgRef.current!);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      saveAs(blob, 'graph-visualization.svg');
    };

    const resetZoom = () => {
      svg.transition().duration(750).call(zoom.transform as any, d3.zoomIdentity);
    };

    // Store functions for external use
    (window as any).graphExportSVG = exportSVG;
    (window as any).graphResetZoom = resetZoom;

    return () => {
      simulation.stop();
      delete (window as any).graphExportSVG;
      delete (window as any).graphResetZoom;
    };
  }, [data, theme, layoutType, showLabels, nodeSize, linkStrength]);

  const stats = useMemo(() => ({
    nodes: data.nodes.length,
    edges: data.edges.length,
    nodeTypes: [...new Set(data.nodes.map(n => n.type))].length,
    edgeTypes: [...new Set(data.edges.map(e => e.type))].length,
  }), [data]);

  return (
    <Paper
      elevation={0}
      sx={{
        height: fullscreen ? '100vh' : 700,
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        backgroundColor: alpha(theme.palette.background.paper, 0.9),
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box>
          <Typography variant="h6">
            {data.metadata?.title || 'Knowledge Graph'}
          </Typography>
          {data.metadata?.description && (
            <Typography variant="body2" color="text.secondary">
              {data.metadata.description}
            </Typography>
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip label={`${stats.nodes} nodes`} size="small" />
          <Chip label={`${stats.edges} edges`} size="small" />
          <Chip label={`${stats.nodeTypes} types`} size="small" variant="outlined" />
        </Box>
      </Box>

      {/* Visualization */}
      <Box ref={containerRef} sx={{ flex: 1, position: 'relative' }}>
        <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

        {/* Controls */}
        <Box
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
          }}
        >
          <Tooltip title="Zoom In">
            <IconButton
              size="small"
              onClick={() => {
                const svg = d3.select(svgRef.current);
                svg.transition().call(d3.zoom().scaleBy as any, 1.3);
              }}
              sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
            >
              <ZoomIn />
            </IconButton>
          </Tooltip>

          <Tooltip title="Zoom Out">
            <IconButton
              size="small"
              onClick={() => {
                const svg = d3.select(svgRef.current);
                svg.transition().call(d3.zoom().scaleBy as any, 0.7);
              }}
              sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
            >
              <ZoomOut />
            </IconButton>
          </Tooltip>

          <Tooltip title="Reset View">
            <IconButton
              size="small"
              onClick={() => (window as any).graphResetZoom?.()}
              sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
            >
              <CenterFocusStrong />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export SVG">
            <IconButton
              size="small"
              onClick={() => (window as any).graphExportSVG?.()}
              sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
            >
              <PhotoCamera />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton
              size="small"
              onClick={() => setShowControls(!showControls)}
              sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
            >
              <Tune />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Settings Panel */}
        {showControls && (
          <Paper
            sx={{
              position: 'absolute',
              top: 16,
              left: 16,
              p: 2,
              width: 250,
              maxHeight: '80%',
              overflow: 'auto',
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Layout Settings
            </Typography>

            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <Select
                value={layoutType}
                onChange={(e) => setLayoutType(e.target.value as any)}
              >
                <MenuItem value="force">Force Layout</MenuItem>
                <MenuItem value="hierarchical">Hierarchical</MenuItem>
                <MenuItem value="circular">Circular</MenuItem>
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Switch
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
              }
              label="Show Labels"
              sx={{ mb: 2 }}
            />

            <Typography variant="body2" gutterBottom>
              Node Size
            </Typography>
            <Slider
              value={nodeSize}
              onChange={(e, value) => setNodeSize(value as number)}
              min={5}
              max={30}
              valueLabelDisplay="auto"
              sx={{ mb: 2 }}
            />

            <Typography variant="body2" gutterBottom>
              Link Strength
            </Typography>
            <Slider
              value={linkStrength}
              onChange={(e, value) => setLinkStrength(value as number)}
              min={0.1}
              max={2}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Paper>
        )}

        {/* Selection Info */}
        {(selectedNode || selectedEdge) && (
          <Paper
            sx={{
              position: 'absolute',
              bottom: 16,
              left: 16,
              p: 2,
              maxWidth: 300,
            }}
          >
            {selectedNode && (
              <>
                <Typography variant="subtitle2" gutterBottom>
                  Selected Node
                </Typography>
                <Typography variant="body2">
                  <strong>ID:</strong> {selectedNode.id}
                </Typography>
                <Typography variant="body2">
                  <strong>Type:</strong> {selectedNode.type}
                </Typography>
                {selectedNode.label && (
                  <Typography variant="body2">
                    <strong>Label:</strong> {selectedNode.label}
                  </Typography>
                )}
                {selectedNode.properties && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Properties:
                    </Typography>
                    <pre style={{ fontSize: '0.75rem', margin: 0 }}>
                      {JSON.stringify(selectedNode.properties, null, 2)}
                    </pre>
                  </Box>
                )}
              </>
            )}

            {selectedEdge && (
              <>
                <Typography variant="subtitle2" gutterBottom>
                  Selected Edge
                </Typography>
                <Typography variant="body2">
                  <strong>Type:</strong> {selectedEdge.type}
                </Typography>
                <Typography variant="body2">
                  <strong>Source:</strong> {selectedEdge.source}
                </Typography>
                <Typography variant="body2">
                  <strong>Target:</strong> {selectedEdge.target}
                </Typography>
                {selectedEdge.weight && (
                  <Typography variant="body2">
                    <strong>Weight:</strong> {selectedEdge.weight}
                  </Typography>
                )}
              </>
            )}
          </Paper>
        )}
      </Box>
    </Paper>
  );
};

export default GraphVisualization;