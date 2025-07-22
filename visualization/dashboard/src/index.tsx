import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { reportWebVitals } from './reportWebVitals';

// Get the root element
const container = document.getElementById('root');

if (!container) {
  throw new Error('Root element not found. Make sure you have a div with id="root" in your HTML.');
}

// Create React root
const root = createRoot(container);

// Render the App
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Performance monitoring
reportWebVitals((metric) => {
  // Log performance metrics in development
  if (process.env.NODE_ENV === 'development') {
    console.log('Web Vital:', metric);
  }
  
  // In production, you might want to send metrics to an analytics service
  // Example: gtag('event', metric.name, { metric_value: metric.value });
});