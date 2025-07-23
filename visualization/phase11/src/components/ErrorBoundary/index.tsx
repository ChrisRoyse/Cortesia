import React from 'react';
import { Result, Button, Card, Typography, Space, Alert, Collapse } from 'antd';
import { ReloadOutlined, BugOutlined, HomeOutlined, CopyOutlined } from '@ant-design/icons';
import { motion } from 'framer-motion';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface ErrorInfo {
  componentStack: string;
  errorBoundary?: string;
  errorBoundaryStack?: string;
}

interface ErrorFallbackProps {
  error: Error;
  errorInfo?: ErrorInfo;
  resetErrorBoundary?: () => void;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  errorInfo,
  resetErrorBoundary,
}) => {
  const handleCopyError = () => {
    const errorText = `
Error: ${error.message}
Stack: ${error.stack}
Component Stack: ${errorInfo?.componentStack || 'N/A'}
Timestamp: ${new Date().toISOString()}
URL: ${window.location.href}
User Agent: ${navigator.userAgent}
    `.trim();

    navigator.clipboard.writeText(errorText).then(() => {
      // Could show a toast here
      console.log('Error details copied to clipboard');
    });
  };

  const handleReportError = () => {
    // In a real application, this would send the error to a reporting service
    console.log('Error would be reported to monitoring service:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      url: window.location.href,
    });
    
    alert('Error report sent! (This is a demo - no actual report was sent)');
  };

  const handleGoHome = () => {
    window.location.href = '/';
  };

  const isNetworkError = error.message.includes('fetch') || error.message.includes('network');
  const isChunkError = error.message.includes('Loading chunk') || error.message.includes('ChunkLoadError');

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        minHeight: '100vh',
        background: 'var(--background-color, #001529)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem',
      }}
    >
      <Card
        style={{
          maxWidth: '800px',
          width: '100%',
          background: 'var(--surface-color, #141414)',
          border: '1px solid var(--border-color, #303030)',
        }}
      >
        <Result
          status="error"
          title="Oops! Something went wrong"
          subTitle={
            isChunkError
              ? "A new version of the application is available. Please refresh the page."
              : isNetworkError
              ? "Network connection issue. Please check your internet connection."
              : "An unexpected error occurred while rendering this component."
          }
          extra={
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Space wrap>
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={resetErrorBoundary || (() => window.location.reload())}
                  size="large"
                >
                  {isChunkError ? 'Refresh Page' : 'Try Again'}
                </Button>
                
                <Button
                  icon={<HomeOutlined />}
                  onClick={handleGoHome}
                  size="large"
                >
                  Go Home
                </Button>
                
                <Button
                  icon={<CopyOutlined />}
                  onClick={handleCopyError}
                  size="large"
                >
                  Copy Error
                </Button>
                
                <Button
                  icon={<BugOutlined />}
                  onClick={handleReportError}
                  size="large"
                >
                  Report Issue
                </Button>
              </Space>

              {/* Error Details */}
              <Alert
                message="Error Details"
                description={
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>Error Message:</Text>
                    <Text code style={{ wordBreak: 'break-word' }}>
                      {error.message}
                    </Text>
                    
                    <Collapse
                      ghost
                      items={[
                        {
                          key: 'stack',
                          label: 'Stack Trace',
                          children: (
                            <pre
                              style={{
                                background: 'var(--background-color, #001529)',
                                padding: '12px',
                                borderRadius: '4px',
                                fontSize: '12px',
                                maxHeight: '300px',
                                overflow: 'auto',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                              }}
                            >
                              {error.stack}
                            </pre>
                          ),
                        },
                        ...(errorInfo?.componentStack
                          ? [
                              {
                                key: 'component',
                                label: 'Component Stack',
                                children: (
                                  <pre
                                    style={{
                                      background: 'var(--background-color, #001529)',
                                      padding: '12px',
                                      borderRadius: '4px',
                                      fontSize: '12px',
                                      maxHeight: '300px',
                                      overflow: 'auto',
                                      whiteSpace: 'pre-wrap',
                                      wordBreak: 'break-word',
                                    }}
                                  >
                                    {errorInfo.componentStack}
                                  </pre>
                                ),
                              },
                            ]
                          : []),
                      ]}
                    />
                  </Space>
                }
                type="error"
                showIcon
                style={{ textAlign: 'left' }}
              />

              {/* Troubleshooting Tips */}
              <Alert
                message="Troubleshooting Tips"
                description={
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {isChunkError && (
                      <Text>
                        • A new version of the application has been deployed. Refreshing the page will load the latest version.
                      </Text>
                    )}
                    {isNetworkError && (
                      <>
                        <Text>• Check your internet connection</Text>
                        <Text>• Verify that the LLMKG server is running</Text>
                        <Text>• Try refreshing the page</Text>
                      </>
                    )}
                    {!isChunkError && !isNetworkError && (
                      <>
                        <Text>• Try refreshing the page</Text>
                        <Text>• Clear your browser cache and cookies</Text>
                        <Text>• Check the browser console for additional errors</Text>
                        <Text>• Contact support if the issue persists</Text>
                      </>
                    )}
                  </Space>
                }
                type="info"
                showIcon
                style={{ textAlign: 'left' }}
              />

              {/* System Information */}
              <Alert
                message="System Information"
                description={
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text>
                      <strong>Timestamp:</strong> {new Date().toLocaleString()}
                    </Text>
                    <Text>
                      <strong>URL:</strong> {window.location.href}
                    </Text>
                    <Text>
                      <strong>User Agent:</strong> {navigator.userAgent}
                    </Text>
                    <Text>
                      <strong>Version:</strong> 2.0.0
                    </Text>
                  </Space>
                }
                type="info"
                showIcon
                style={{ textAlign: 'left' }}
              />
            </Space>
          }
        />
      </Card>
    </motion.div>
  );
};

// React Error Boundary component
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{
    fallback?: React.ComponentType<ErrorFallbackProps>;
    onError?: (error: Error, errorInfo: ErrorInfo) => void;
  }>,
  ErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error Boundary caught an error:', error);
    console.error('Error info:', errorInfo);

    this.setState({
      error,
      errorInfo: {
        componentStack: errorInfo.componentStack,
      },
    });

    // Call the onError callback if provided
    this.props.onError?.(error, {
      componentStack: errorInfo.componentStack,
    });

    // In production, report to error monitoring service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry, LogRocket, etc.
      console.log('Would report error to monitoring service:', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
      });
    }
  }

  render() {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback || ErrorFallback;
      
      return (
        <FallbackComponent
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          resetErrorBoundary={() => {
            this.setState({ hasError: false, error: undefined, errorInfo: undefined });
          }}
        />
      );
    }

    return this.props.children;
  }
}