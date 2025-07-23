import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button, Alert } from '@mui/material';
import { RefreshRounded, BugReportRounded } from '@mui/icons-material';

interface Props {
  children: ReactNode;
  componentName?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class VisualizationErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Visualization Error Boundary caught an error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  override render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ 
          width: '100%', 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center', 
          justifyContent: 'center',
          p: 3,
          minHeight: 200
        }}>
          <Alert 
            severity="error" 
            icon={<BugReportRounded />}
            sx={{ mb: 2, width: '100%', maxWidth: 400 }}
          >
            <Typography variant="h6" gutterBottom>
              Visualization Error
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              {this.props.componentName ? `${this.props.componentName} ` : 'This visualization '}
              encountered an error and could not render.
            </Typography>
            {this.state.error && (
              <Typography variant="caption" sx={{ 
                display: 'block',
                fontFamily: 'monospace',
                backgroundColor: 'rgba(0,0,0,0.1)',
                p: 1,
                borderRadius: 1,
                mt: 1
              }}>
                {this.state.error.message}
              </Typography>
            )}
          </Alert>
          
          <Button
            variant="outlined"
            startIcon={<RefreshRounded />}
            onClick={this.handleReset}
            sx={{ mt: 1 }}
          >
            Try Again
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}