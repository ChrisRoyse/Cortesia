import React, { useState } from 'react';
import { Box, Container, Paper, Tab, Tabs, Typography } from '@mui/material';
import StatusDashboard from './StatusDashboard';
import { useAllToolsStatus, useProblematicToolsStatus } from '../../hooks/useToolStatus';
import { MCPTool } from '../../types';

/**
 * Example implementation of the Status Monitoring System
 * 
 * This component demonstrates how to use the live status monitoring
 * system with different configurations and use cases.
 */
const StatusDashboardExample: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);

  // Monitor all tools with custom refresh interval
  const allToolsStatus = useAllToolsStatus({
    refreshInterval: 30000, // 30 seconds
    onStatusChange: (toolId, oldStatus, newStatus) => {
      console.log(`Tool ${toolId} changed from ${oldStatus} to ${newStatus}`);
    },
    onAlert: (toolId, status, message) => {
      console.warn(`Alert for tool ${toolId}: ${message}`);
    }
  });

  // Monitor only problematic tools
  const problematicToolsStatus = useProblematicToolsStatus({
    refreshInterval: 15000, // Check problematic tools more frequently
    onAlert: (toolId, status, message) => {
      // Could trigger notifications, send emails, etc.
      console.error(`Critical alert for tool ${toolId}: ${message}`);
    }
  });

  // Get statistics
  const stats = allToolsStatus.getStats();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  const handleToolSelect = (tool: MCPTool) => {
    setSelectedTool(tool);
    console.log('Selected tool:', tool);
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          MCP Tool Status Monitoring
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph>
          Live monitoring system for all discovered MCP tools with real-time health checks,
          performance metrics, and alert generation.
        </Typography>

        {/* Statistics Summary */}
        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            System Overview
          </Typography>
          <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Average Response Time
              </Typography>
              <Typography variant="h4">
                {stats.averageResponseTime.toFixed(0)}ms
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                System Error Rate
              </Typography>
              <Typography variant="h4">
                {(stats.errorRate * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Overall Availability
              </Typography>
              <Typography variant="h4">
                {stats.availability.toFixed(1)}%
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Last Update
              </Typography>
              <Typography variant="h4">
                {allToolsStatus.lastUpdate?.toLocaleTimeString() || 'N/A'}
              </Typography>
            </Box>
          </Box>
        </Paper>

        {/* Tabbed Views */}
        <Paper elevation={2}>
          <Tabs value={selectedTab} onChange={handleTabChange}>
            <Tab label="All Tools Dashboard" />
            <Tab label="Problematic Tools" />
            <Tab label="Category View" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {selectedTab === 0 && (
              <StatusDashboard
                refreshInterval={30000}
                showHistory={true}
                onToolSelect={handleToolSelect}
              />
            )}

            {selectedTab === 1 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Tools Requiring Attention
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Monitoring {problematicToolsStatus.statuses.size} problematic tools
                  with increased check frequency (every 15 seconds).
                </Typography>
                <StatusDashboard
                  tools={Array.from(problematicToolsStatus.statuses.values()).map(status => ({
                    // This would come from the actual tool data
                    id: 'tool-id',
                    name: 'Tool Name',
                    status
                  } as any))}
                  refreshInterval={15000}
                  showHistory={true}
                  onToolSelect={handleToolSelect}
                />
              </Box>
            )}

            {selectedTab === 2 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Tools by Category
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  View tool health organized by functional categories.
                </Typography>
                {/* Category-specific dashboards would go here */}
                <StatusDashboard
                  refreshInterval={30000}
                  showHistory={false}
                  onToolSelect={handleToolSelect}
                />
              </Box>
            )}
          </Box>
        </Paper>

        {/* Selected Tool Details */}
        {selectedTool && (
          <Paper elevation={2} sx={{ mt: 3, p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Selected Tool: {selectedTool.name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              ID: {selectedTool.id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Category: {selectedTool.category}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Status: {selectedTool.status.health}
            </Typography>
            {/* Additional tool details and actions would go here */}
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default StatusDashboardExample;