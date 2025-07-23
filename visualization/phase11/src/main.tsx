import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import { ConfigProvider, theme } from 'antd';
import { ErrorBoundary } from 'react-error-boundary';

import App from './App';
import { store } from './stores';
import { ErrorFallback } from './components/ErrorBoundary';
import { performanceMonitor } from './utils/performance';
import { initializeApp } from './config/initialization';

// Import global styles
import './styles/globals.css';
import './styles/antd-overrides.css';

// Initialize performance monitoring
if (import.meta.env.PROD) {
  performanceMonitor.init();
}

// Initialize application
initializeApp();

// Antd theme configuration
const antdTheme = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#1890ff',
    colorBgContainer: '#001529',
    colorBgElevated: '#141414',
    colorText: '#ffffff',
    colorTextSecondary: 'rgba(255, 255, 255, 0.65)',
    borderRadius: 6,
    wireframe: false,
  },
  components: {
    Layout: {
      bodyBg: '#001529',
      headerBg: '#001529',
      siderBg: '#001529',
    },
    Menu: {
      darkItemBg: 'transparent',
      darkItemSelectedBg: '#1890ff',
      darkItemHoverBg: 'rgba(24, 144, 255, 0.2)',
    },
    Card: {
      colorBgContainer: '#141414',
    },
    Table: {
      colorBgContainer: '#141414',
    },
  },
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

root.render(
  <React.StrictMode>
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, errorInfo) => {
        console.error('Application Error:', error);
        console.error('Error Info:', errorInfo);
        
        // Send error to monitoring service in production
        if (import.meta.env.PROD) {
          // Performance monitor would send this to analytics
          performanceMonitor.reportError(error, errorInfo);
        }
      }}
    >
      <Provider store={store}>
        <ConfigProvider theme={antdTheme}>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ConfigProvider>
      </Provider>
    </ErrorBoundary>
  </React.StrictMode>
);

// Register service worker for PWA functionality
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}

// Performance monitoring for Core Web Vitals
if (import.meta.env.PROD) {
  import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
    getCLS(performanceMonitor.onCLS);
    getFID(performanceMonitor.onFID);
    getFCP(performanceMonitor.onFCP);
    getLCP(performanceMonitor.onLCP);
    getTTFB(performanceMonitor.onTTFB);
  });
}