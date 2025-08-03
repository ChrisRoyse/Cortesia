# Micro-Phase 9.22: Responsive CSS Design Implementation

## Objective
Implement comprehensive responsive CSS design with modern styling, dark/light themes, animations, and accessibility features for the CortexKG web interface across all device sizes.

## Prerequisites
- Completed micro-phase 9.21 (HTML structure)
- HTML template structure finalized
- Accessibility requirements defined
- Design system decisions made

## Task Description
Create a complete CSS design system with responsive layouts, theme support, smooth animations, and accessibility-first styling that provides an excellent user experience across desktop, tablet, and mobile devices.

## Specific Actions

1. **Create main CSS framework**:
   ```css
   /* public/styles/main.css */
   
   /* CSS Custom Properties (Design Tokens) */
   :root {
     /* Colors - Light Theme */
     --color-primary: #1a365d;
     --color-primary-light: #2c5282;
     --color-primary-dark: #153e75;
     --color-secondary: #319795;
     --color-secondary-light: #4fd1c7;
     --color-secondary-dark: #2c7a7b;
     
     --color-background: #ffffff;
     --color-background-alt: #f7fafc;
     --color-surface: #ffffff;
     --color-surface-raised: #ffffff;
     
     --color-text: #1a202c;
     --color-text-secondary: #4a5568;
     --color-text-muted: #718096;
     --color-text-inverse: #ffffff;
     
     --color-border: #e2e8f0;
     --color-border-light: #edf2f7;
     --color-border-dark: #cbd5e0;
     
     --color-success: #38a169;
     --color-success-light: #48bb78;
     --color-warning: #d69e2e;
     --color-warning-light: #ecc94b;
     --color-error: #e53e3e;
     --color-error-light: #f56565;
     --color-info: #3182ce;
     --color-info-light: #4299e1;
     
     /* Spacing */
     --spacing-xs: 0.25rem;
     --spacing-sm: 0.5rem;
     --spacing-md: 1rem;
     --spacing-lg: 1.5rem;
     --spacing-xl: 2rem;
     --spacing-2xl: 3rem;
     --spacing-3xl: 4rem;
     
     /* Typography */
     --font-family-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
     --font-family-mono: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
     
     --font-size-xs: 0.75rem;
     --font-size-sm: 0.875rem;
     --font-size-base: 1rem;
     --font-size-lg: 1.125rem;
     --font-size-xl: 1.25rem;
     --font-size-2xl: 1.5rem;
     --font-size-3xl: 1.875rem;
     --font-size-4xl: 2.25rem;
     
     --font-weight-normal: 400;
     --font-weight-medium: 500;
     --font-weight-semibold: 600;
     --font-weight-bold: 700;
     
     --line-height-tight: 1.25;
     --line-height-normal: 1.5;
     --line-height-relaxed: 1.75;
     
     /* Shadows */
     --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
     --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
     --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
     --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
     
     /* Border Radius */
     --radius-sm: 0.125rem;
     --radius-md: 0.375rem;
     --radius-lg: 0.5rem;
     --radius-xl: 0.75rem;
     --radius-2xl: 1rem;
     --radius-full: 9999px;
     
     /* Transitions */
     --transition-fast: 150ms ease-in-out;
     --transition-normal: 250ms ease-in-out;
     --transition-slow: 350ms ease-in-out;
     
     /* Z-Index Scale */
     --z-dropdown: 1000;
     --z-sticky: 1020;
     --z-fixed: 1030;
     --z-modal-backdrop: 1040;
     --z-modal: 1050;
     --z-popover: 1060;
     --z-tooltip: 1070;
     --z-toast: 1080;
   }

   /* Dark Theme */
   [data-theme="dark"] {
     --color-background: #1a202c;
     --color-background-alt: #2d3748;
     --color-surface: #2d3748;
     --color-surface-raised: #4a5568;
     
     --color-text: #ffffff;
     --color-text-secondary: #cbd5e0;
     --color-text-muted: #a0aec0;
     --color-text-inverse: #1a202c;
     
     --color-border: #4a5568;
     --color-border-light: #2d3748;
     --color-border-dark: #718096;
   }

   /* Auto theme based on system preference */
   @media (prefers-color-scheme: dark) {
     :root:not([data-theme]) {
       --color-background: #1a202c;
       --color-background-alt: #2d3748;
       --color-surface: #2d3748;
       --color-surface-raised: #4a5568;
       
       --color-text: #ffffff;
       --color-text-secondary: #cbd5e0;
       --color-text-muted: #a0aec0;
       --color-text-inverse: #1a202c;
       
       --color-border: #4a5568;
       --color-border-light: #2d3748;
       --color-border-dark: #718096;
     }
   }

   /* Reset and Base Styles */
   *,
   *::before,
   *::after {
     box-sizing: border-box;
   }

   * {
     margin: 0;
     padding: 0;
   }

   html {
     font-size: 16px;
     scroll-behavior: smooth;
     height: 100%;
   }

   body {
     font-family: var(--font-family-sans);
     font-size: var(--font-size-base);
     line-height: var(--line-height-normal);
     color: var(--color-text);
     background-color: var(--color-background);
     min-height: 100vh;
     display: flex;
     flex-direction: column;
     -webkit-font-smoothing: antialiased;
     -moz-osx-font-smoothing: grayscale;
   }

   /* Focus Management */
   *:focus {
     outline: 2px solid var(--color-primary);
     outline-offset: 2px;
   }

   .focus-visible-only:focus:not(:focus-visible) {
     outline: none;
   }

   /* Skip Link */
   .skip-link {
     position: absolute;
     top: -40px;
     left: 6px;
     background: var(--color-primary);
     color: var(--color-text-inverse);
     padding: 8px;
     text-decoration: none;
     border-radius: var(--radius-md);
     z-index: var(--z-tooltip);
     transition: top var(--transition-fast);
   }

   .skip-link:focus {
     top: 6px;
   }

   /* Loading Screen */
   .loading-screen {
     position: fixed;
     top: 0;
     left: 0;
     right: 0;
     bottom: 0;
     background: var(--color-background);
     display: flex;
     align-items: center;
     justify-content: center;
     z-index: var(--z-modal);
     transition: opacity var(--transition-slow), visibility var(--transition-slow);
   }

   .loading-screen[aria-hidden="true"] {
     opacity: 0;
     visibility: hidden;
   }

   .loading-spinner {
     text-align: center;
     max-width: 300px;
   }

   .spinner-circle {
     width: 64px;
     height: 64px;
     border: 4px solid var(--color-border);
     border-top: 4px solid var(--color-primary);
     border-radius: var(--radius-full);
     animation: spin 1s linear infinite;
     margin: 0 auto var(--spacing-lg);
   }

   @keyframes spin {
     0% { transform: rotate(0deg); }
     100% { transform: rotate(360deg); }
   }

   .loading-text h2 {
     font-size: var(--font-size-2xl);
     font-weight: var(--font-weight-semibold);
     margin-bottom: var(--spacing-sm);
     color: var(--color-text);
   }

   .loading-text p {
     color: var(--color-text-secondary);
     margin-bottom: var(--spacing-lg);
   }

   .progress-bar {
     width: 100%;
     height: 4px;
     background: var(--color-border);
     border-radius: var(--radius-full);
     overflow: hidden;
   }

   .progress-fill {
     height: 100%;
     background: var(--color-primary);
     border-radius: var(--radius-full);
     transition: width var(--transition-normal);
   }

   /* App Container */
   .app-container {
     flex: 1;
     display: flex;
     flex-direction: column;
     min-height: 100vh;
     transition: opacity var(--transition-slow);
   }

   .app-container[aria-hidden="true"] {
     opacity: 0;
   }

   /* Header */
   .app-header {
     background: var(--color-surface);
     border-bottom: 1px solid var(--color-border);
     box-shadow: var(--shadow-sm);
     position: sticky;
     top: 0;
     z-index: var(--z-sticky);
   }

   .header-content {
     display: flex;
     align-items: center;
     justify-content: space-between;
     padding: var(--spacing-md) var(--spacing-lg);
     max-width: 1200px;
     margin: 0 auto;
   }

   .logo-section {
     display: flex;
     align-items: center;
     gap: var(--spacing-sm);
   }

   .logo-icon {
     width: 32px;
     height: 32px;
     fill: var(--color-primary);
   }

   .app-title {
     font-size: var(--font-size-xl);
     font-weight: var(--font-weight-bold);
     color: var(--color-primary);
   }

   /* Navigation */
   .main-navigation {
     flex: 1;
     display: flex;
     justify-content: center;
     max-width: 600px;
   }

   .nav-list {
     display: flex;
     list-style: none;
     gap: var(--spacing-xs);
     background: var(--color-background-alt);
     padding: var(--spacing-xs);
     border-radius: var(--radius-lg);
   }

   .nav-button {
     background: none;
     border: none;
     padding: var(--spacing-sm) var(--spacing-lg);
     border-radius: var(--radius-md);
     font-size: var(--font-size-sm);
     font-weight: var(--font-weight-medium);
     color: var(--color-text-secondary);
     cursor: pointer;
     transition: all var(--transition-fast);
     position: relative;
   }

   .nav-button:hover {
     color: var(--color-text);
     background: var(--color-surface);
   }

   .nav-button.active,
   .nav-button[aria-pressed="true"] {
     background: var(--color-primary);
     color: var(--color-text-inverse);
     box-shadow: var(--shadow-sm);
   }

   /* Header Actions */
   .header-actions {
     display: flex;
     align-items: center;
     gap: var(--spacing-md);
   }

   .icon-button {
     background: none;
     border: none;
     padding: var(--spacing-sm);
     border-radius: var(--radius-md);
     color: var(--color-text-secondary);
     cursor: pointer;
     transition: all var(--transition-fast);
     display: flex;
     align-items: center;
     justify-content: center;
   }

   .icon-button:hover {
     background: var(--color-background-alt);
     color: var(--color-text);
   }

   .icon-button svg {
     width: 20px;
     height: 20px;
     fill: currentColor;
   }

   .status-indicator {
     display: flex;
     align-items: center;
     gap: var(--spacing-sm);
   }

   .status-dot {
     width: 8px;
     height: 8px;
     border-radius: var(--radius-full);
     transition: background-color var(--transition-fast);
   }

   .status-dot.status-connecting {
     background: var(--color-warning);
     animation: pulse 2s infinite;
   }

   .status-dot.status-connected {
     background: var(--color-success);
   }

   .status-dot.status-error {
     background: var(--color-error);
   }

   @keyframes pulse {
     0%, 100% { opacity: 1; }
     50% { opacity: 0.5; }
   }

   .status-text {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
   }

   /* Main Content */
   .main-content {
     flex: 1;
     padding: var(--spacing-lg);
     max-width: 1200px;
     margin: 0 auto;
     width: 100%;
   }

   /* Tab System */
   .tab-content {
     display: none;
     opacity: 0;
     transform: translateY(20px);
     transition: all var(--transition-normal);
   }

   .tab-content.active {
     display: block;
     opacity: 1;
     transform: translateY(0);
   }

   .tab-content[aria-hidden="false"] {
     display: block;
     opacity: 1;
     transform: translateY(0);
   }

   /* Section Headers */
   .section-header {
     margin-bottom: var(--spacing-xl);
     text-align: center;
   }

   .section-header h2 {
     font-size: var(--font-size-3xl);
     font-weight: var(--font-weight-bold);
     color: var(--color-text);
     margin-bottom: var(--spacing-sm);
   }

   .section-header p {
     font-size: var(--font-size-lg);
     color: var(--color-text-secondary);
   }

   /* Form Styles */
   .input-form {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-xl);
     padding: var(--spacing-xl);
     box-shadow: var(--shadow-sm);
     margin-bottom: var(--spacing-xl);
   }

   .form-group {
     margin-bottom: var(--spacing-lg);
   }

   .form-group.inline {
     display: flex;
     align-items: center;
     gap: var(--spacing-md);
     margin-bottom: var(--spacing-md);
   }

   .form-label {
     display: block;
     font-weight: var(--font-weight-medium);
     color: var(--color-text);
     margin-bottom: var(--spacing-sm);
     font-size: var(--font-size-sm);
   }

   .form-group.inline .form-label {
     margin-bottom: 0;
     min-width: 120px;
   }

   .required {
     color: var(--color-error);
     margin-left: var(--spacing-xs);
   }

   .form-input,
   .form-textarea,
   .form-select {
     width: 100%;
     padding: var(--spacing-md);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-md);
     font-size: var(--font-size-base);
     font-family: inherit;
     background: var(--color-background);
     color: var(--color-text);
     transition: all var(--transition-fast);
   }

   .form-input:focus,
   .form-textarea:focus,
   .form-select:focus {
     border-color: var(--color-primary);
     box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
     outline: none;
   }

   .form-textarea {
     resize: vertical;
     min-height: 120px;
   }

   .form-range {
     width: 100%;
     height: 6px;
     border-radius: var(--radius-full);
     background: var(--color-border);
     outline: none;
     -webkit-appearance: none;
   }

   .form-range::-webkit-slider-thumb {
     appearance: none;
     width: 20px;
     height: 20px;
     border-radius: var(--radius-full);
     background: var(--color-primary);
     cursor: pointer;
     box-shadow: var(--shadow-sm);
   }

   .form-range::-moz-range-thumb {
     width: 20px;
     height: 20px;
     border-radius: var(--radius-full);
     background: var(--color-primary);
     cursor: pointer;
     border: none;
     box-shadow: var(--shadow-sm);
   }

   .form-help {
     font-size: var(--font-size-sm);
     color: var(--color-text-muted);
     margin-top: var(--spacing-xs);
   }

   .form-error {
     font-size: var(--font-size-sm);
     color: var(--color-error);
     margin-top: var(--spacing-xs);
     display: none;
   }

   .form-error:not(:empty) {
     display: block;
   }

   .character-count {
     font-size: var(--font-size-xs);
     color: var(--color-text-muted);
     text-align: right;
     margin-top: var(--spacing-xs);
   }

   .range-value {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
     font-weight: var(--font-weight-medium);
     min-width: 40px;
     text-align: center;
   }

   /* Buttons */
   .primary-button,
   .secondary-button {
     position: relative;
     display: inline-flex;
     align-items: center;
     justify-content: center;
     gap: var(--spacing-sm);
     padding: var(--spacing-md) var(--spacing-xl);
     border-radius: var(--radius-md);
     font-size: var(--font-size-base);
     font-weight: var(--font-weight-medium);
     text-decoration: none;
     cursor: pointer;
     transition: all var(--transition-fast);
     border: 1px solid transparent;
     min-height: 44px; /* Touch target size */
   }

   .primary-button {
     background: var(--color-primary);
     color: var(--color-text-inverse);
     border-color: var(--color-primary);
   }

   .primary-button:hover {
     background: var(--color-primary-light);
     border-color: var(--color-primary-light);
     transform: translateY(-1px);
     box-shadow: var(--shadow-md);
   }

   .primary-button:active {
     transform: translateY(0);
     box-shadow: var(--shadow-sm);
   }

   .secondary-button {
     background: transparent;
     color: var(--color-text);
     border-color: var(--color-border-dark);
   }

   .secondary-button:hover {
     background: var(--color-background-alt);
     border-color: var(--color-border-dark);
   }

   .primary-button:disabled,
   .secondary-button:disabled {
     opacity: 0.5;
     cursor: not-allowed;
     transform: none;
   }

   .button-spinner {
     width: 16px;
     height: 16px;
     border: 2px solid transparent;
     border-top: 2px solid currentColor;
     border-radius: var(--radius-full);
     animation: spin 1s linear infinite;
     display: none;
   }

   .primary-button:disabled .button-spinner {
     display: block;
   }

   .primary-button:disabled .button-text {
     opacity: 0.7;
   }

   .form-actions {
     display: flex;
     gap: var(--spacing-md);
     justify-content: flex-start;
     margin-top: var(--spacing-xl);
   }

   /* Results Section */
   .results-section {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-xl);
     padding: var(--spacing-xl);
     min-height: 200px;
     display: flex;
     flex-direction: column;
   }

   .results-section:empty::before {
     content: "Results will appear here...";
     color: var(--color-text-muted);
     font-style: italic;
     text-align: center;
     margin: auto;
   }

   /* Result Cards */
   .result-card {
     background: var(--color-surface-raised);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     padding: var(--spacing-lg);
     margin-bottom: var(--spacing-md);
     box-shadow: var(--shadow-sm);
     transition: all var(--transition-fast);
   }

   .result-card:hover {
     box-shadow: var(--shadow-md);
     transform: translateY(-2px);
   }

   .result-card.success {
     border-left: 4px solid var(--color-success);
   }

   .result-card.error {
     border-left: 4px solid var(--color-error);
   }

   .result-header {
     display: flex;
     justify-content: space-between;
     align-items: center;
     margin-bottom: var(--spacing-md);
   }

   .result-header h3 {
     font-size: var(--font-size-lg);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
   }

   .result-timestamp {
     font-size: var(--font-size-sm);
     color: var(--color-text-muted);
   }

   .result-body {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
     gap: var(--spacing-md);
   }

   .result-metric {
     display: flex;
     justify-content: space-between;
     align-items: center;
     padding: var(--spacing-sm) 0;
   }

   .metric-label {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
     font-weight: var(--font-weight-medium);
   }

   .metric-value {
     font-size: var(--font-size-base);
     color: var(--color-text);
     font-weight: var(--font-weight-semibold);
   }

   /* Query Results */
   .query-result-item {
     display: flex;
     align-items: flex-start;
     gap: var(--spacing-md);
     padding: var(--spacing-lg);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     margin-bottom: var(--spacing-md);
     background: var(--color-surface-raised);
     transition: all var(--transition-fast);
     cursor: pointer;
   }

   .query-result-item:hover {
     box-shadow: var(--shadow-md);
     transform: translateY(-1px);
   }

   .query-result-item:focus {
     box-shadow: var(--shadow-lg);
   }

   .result-rank {
     background: var(--color-primary);
     color: var(--color-text-inverse);
     width: 32px;
     height: 32px;
     border-radius: var(--radius-full);
     display: flex;
     align-items: center;
     justify-content: center;
     font-size: var(--font-size-sm);
     font-weight: var(--font-weight-bold);
     flex-shrink: 0;
   }

   .result-content {
     flex: 1;
   }

   .result-text {
     font-size: var(--font-size-base);
     color: var(--color-text);
     margin-bottom: var(--spacing-sm);
     line-height: var(--line-height-relaxed);
   }

   .result-meta {
     display: flex;
     gap: var(--spacing-lg);
     font-size: var(--font-size-sm);
     color: var(--color-text-muted);
   }

   .relevance-score {
     font-weight: var(--font-weight-medium);
   }

   /* Responsive Design */
   @media (max-width: 768px) {
     .header-content {
       padding: var(--spacing-sm) var(--spacing-md);
       flex-direction: column;
       gap: var(--spacing-md);
     }

     .main-navigation {
       order: -1;
       max-width: none;
     }

     .nav-list {
       width: 100%;
       justify-content: space-around;
     }

     .nav-button {
       flex: 1;
       text-align: center;
     }

     .main-content {
       padding: var(--spacing-md);
     }

     .section-header h2 {
       font-size: var(--font-size-2xl);
     }

     .input-form {
       padding: var(--spacing-lg);
     }

     .form-actions {
       flex-direction: column;
     }

     .result-body {
       grid-template-columns: 1fr;
     }

     .query-result-item {
       flex-direction: column;
       gap: var(--spacing-sm);
     }

     .result-rank {
       align-self: flex-start;
     }
   }

   @media (max-width: 480px) {
     .header-content {
       padding: var(--spacing-sm);
     }

     .main-content {
       padding: var(--spacing-sm);
     }

     .input-form {
       padding: var(--spacing-md);
     }

     .section-header h2 {
       font-size: var(--font-size-xl);
     }
   }

   /* Reduced Motion */
   @media (prefers-reduced-motion: reduce) {
     *,
     *::before,
     *::after {
       animation-duration: 0.01ms !important;
       animation-iteration-count: 1 !important;
       transition-duration: 0.01ms !important;
       scroll-behavior: auto !important;
     }

     .spinner-circle,
     .button-spinner {
       animation: none;
     }

     .status-dot.status-connecting {
       animation: none;
     }
   }

   /* High Contrast Mode */
   @media (prefers-contrast: high) {
     :root {
       --color-border: #000000;
       --color-border-light: #000000;
       --color-border-dark: #000000;
     }

     .form-input:focus,
     .form-textarea:focus,
     .form-select:focus {
       border-width: 2px;
     }
   }

   /* Print Styles */
   @media print {
     .app-header,
     .app-footer,
     .form-actions,
     .loading-screen {
       display: none !important;
     }

     .main-content {
       max-width: none;
       padding: 0;
     }

     .result-card,
     .query-result-item {
       break-inside: avoid;
       box-shadow: none;
       border: 1px solid #000;
     }
   }
   ```

2. **Create component-specific styles**:
   ```css
   /* public/styles/components.css */
   
   /* Modal Styles */
   .modal {
     position: fixed;
     top: 0;
     left: 0;
     right: 0;
     bottom: 0;
     z-index: var(--z-modal);
     display: flex;
     align-items: center;
     justify-content: center;
     padding: var(--spacing-lg);
     opacity: 0;
     visibility: hidden;
     transition: all var(--transition-normal);
   }

   .modal[aria-hidden="false"] {
     opacity: 1;
     visibility: visible;
   }

   .modal-backdrop {
     position: absolute;
     top: 0;
     left: 0;
     right: 0;
     bottom: 0;
     background: rgba(0, 0, 0, 0.5);
     backdrop-filter: blur(4px);
   }

   .modal-content {
     position: relative;
     background: var(--color-surface);
     border-radius: var(--radius-xl);
     box-shadow: var(--shadow-xl);
     max-width: 500px;
     width: 100%;
     max-height: 90vh;
     overflow: hidden;
     transform: scale(0.95);
     transition: transform var(--transition-normal);
   }

   .modal[aria-hidden="false"] .modal-content {
     transform: scale(1);
   }

   .modal-header {
     display: flex;
     align-items: center;
     justify-content: space-between;
     padding: var(--spacing-lg);
     border-bottom: 1px solid var(--color-border);
   }

   .modal-header h2 {
     font-size: var(--font-size-xl);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
   }

   .modal-close {
     background: none;
     border: none;
     padding: var(--spacing-sm);
     border-radius: var(--radius-md);
     color: var(--color-text-muted);
     cursor: pointer;
     transition: all var(--transition-fast);
   }

   .modal-close:hover {
     background: var(--color-background-alt);
     color: var(--color-text);
   }

   .modal-close svg {
     width: 20px;
     height: 20px;
     stroke: currentColor;
     stroke-width: 2;
     fill: none;
   }

   .modal-body {
     padding: var(--spacing-lg);
     max-height: 60vh;
     overflow-y: auto;
   }

   .modal-footer {
     display: flex;
     gap: var(--spacing-md);
     justify-content: flex-end;
     padding: var(--spacing-lg);
     border-top: 1px solid var(--color-border);
   }

   /* Settings Styles */
   .settings-section {
     margin-bottom: var(--spacing-xl);
   }

   .settings-section h3 {
     font-size: var(--font-size-lg);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
     margin-bottom: var(--spacing-lg);
     padding-bottom: var(--spacing-sm);
     border-bottom: 1px solid var(--color-border);
   }

   .setting-item {
     margin-bottom: var(--spacing-lg);
   }

   .setting-label {
     display: flex;
     align-items: center;
     gap: var(--spacing-sm);
     font-weight: var(--font-weight-medium);
     color: var(--color-text);
     cursor: pointer;
   }

   .setting-description {
     font-size: var(--font-size-sm);
     color: var(--color-text-muted);
     margin-top: var(--spacing-xs);
     margin-left: 24px;
   }

   .setting-input,
   .setting-select {
     width: 100%;
     max-width: 200px;
     padding: var(--spacing-sm) var(--spacing-md);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-md);
     font-size: var(--font-size-sm);
     background: var(--color-background);
     color: var(--color-text);
   }

   /* Toast Notifications */
   .toast-container {
     position: fixed;
     top: var(--spacing-lg);
     right: var(--spacing-lg);
     z-index: var(--z-toast);
     display: flex;
     flex-direction: column;
     gap: var(--spacing-sm);
     max-width: 400px;
   }

   .toast {
     display: flex;
     align-items: flex-start;
     gap: var(--spacing-md);
     padding: var(--spacing-lg);
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     box-shadow: var(--shadow-lg);
     transform: translateX(100%);
     opacity: 0;
     animation: slideInToast var(--transition-normal) ease-out forwards;
   }

   .toast.toast-success {
     border-left: 4px solid var(--color-success);
   }

   .toast.toast-warning {
     border-left: 4px solid var(--color-warning);
   }

   .toast.toast-error,
   .toast.toast-high,
   .toast.toast-critical {
     border-left: 4px solid var(--color-error);
   }

   .toast.toast-info,
   .toast.toast-low,
   .toast.toast-medium {
     border-left: 4px solid var(--color-info);
   }

   @keyframes slideInToast {
     to {
       transform: translateX(0);
       opacity: 1;
     }
   }

   .toast-icon {
     flex-shrink: 0;
     width: 20px;
     height: 20px;
   }

   .toast-icon svg {
     width: 100%;
     height: 100%;
     fill: currentColor;
   }

   .toast-content {
     flex: 1;
   }

   .toast-title {
     font-size: var(--font-size-sm);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
     margin-bottom: var(--spacing-xs);
   }

   .toast-message {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
   }

   .toast-close {
     background: none;
     border: none;
     padding: var(--spacing-xs);
     color: var(--color-text-muted);
     cursor: pointer;
     border-radius: var(--radius-sm);
     transition: all var(--transition-fast);
   }

   .toast-close:hover {
     background: var(--color-background-alt);
     color: var(--color-text);
   }

   .toast-close svg {
     width: 16px;
     height: 16px;
     stroke: currentColor;
     stroke-width: 2;
     fill: none;
   }

   /* Visualization Styles */
   .visualization-interface {
     display: flex;
     flex-direction: column;
     gap: var(--spacing-lg);
     height: 70vh;
   }

   .viz-controls {
     display: flex;
     align-items: center;
     gap: var(--spacing-xl);
     padding: var(--spacing-lg);
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     flex-wrap: wrap;
   }

   .control-group {
     display: flex;
     align-items: center;
     gap: var(--spacing-md);
   }

   .control-label {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
     font-weight: var(--font-weight-medium);
     min-width: 80px;
   }

   .visualization-container {
     position: relative;
     flex: 1;
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     overflow: hidden;
   }

   .cortical-canvas {
     width: 100%;
     height: 100%;
     display: block;
     cursor: crosshair;
   }

   .viz-overlay {
     position: absolute;
     top: 0;
     left: 0;
     right: 0;
     bottom: 0;
     pointer-events: none;
   }

   .info-panel {
     position: absolute;
     top: var(--spacing-lg);
     left: var(--spacing-lg);
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-md);
     padding: var(--spacing-md);
     box-shadow: var(--shadow-md);
     max-width: 300px;
     pointer-events: auto;
   }

   .info-panel h3 {
     font-size: var(--font-size-base);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
     margin-bottom: var(--spacing-sm);
   }

   .info-panel p {
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
   }

   /* Monitor Styles */
   .metrics-grid {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
     gap: var(--spacing-lg);
     margin-bottom: var(--spacing-xl);
   }

   .metric-card {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     padding: var(--spacing-lg);
     text-align: center;
     box-shadow: var(--shadow-sm);
     transition: all var(--transition-fast);
   }

   .metric-card:hover {
     box-shadow: var(--shadow-md);
     transform: translateY(-2px);
   }

   .metric-title {
     font-size: var(--font-size-sm);
     font-weight: var(--font-weight-medium);
     color: var(--color-text-secondary);
     margin-bottom: var(--spacing-sm);
     text-transform: uppercase;
     letter-spacing: 0.05em;
   }

   .metric-card .metric-value {
     font-size: var(--font-size-3xl);
     font-weight: var(--font-weight-bold);
     color: var(--color-primary);
     margin-bottom: var(--spacing-sm);
   }

   .metric-subtitle {
     font-size: var(--font-size-xs);
     color: var(--color-text-muted);
   }

   .monitor-charts {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
     gap: var(--spacing-lg);
     margin-bottom: var(--spacing-xl);
   }

   .chart-container {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     padding: var(--spacing-lg);
     box-shadow: var(--shadow-sm);
   }

   .chart-container h3 {
     font-size: var(--font-size-lg);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
     margin-bottom: var(--spacing-lg);
   }

   .performance-chart {
     width: 100%;
     height: 200px;
     border-radius: var(--radius-md);
   }

   /* System Logs */
   .system-logs {
     background: var(--color-surface);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-lg);
     padding: var(--spacing-lg);
     box-shadow: var(--shadow-sm);
   }

   .system-logs h3 {
     font-size: var(--font-size-lg);
     font-weight: var(--font-weight-semibold);
     color: var(--color-text);
     margin-bottom: var(--spacing-lg);
   }

   .log-container {
     background: var(--color-background);
     border: 1px solid var(--color-border);
     border-radius: var(--radius-md);
     padding: var(--spacing-md);
     height: 300px;
     overflow-y: auto;
     font-family: var(--font-family-mono);
     margin-bottom: var(--spacing-lg);
   }

   .log-entry {
     display: flex;
     gap: var(--spacing-md);
     padding: var(--spacing-xs) 0;
     border-bottom: 1px solid var(--color-border-light);
     font-size: var(--font-size-sm);
   }

   .log-timestamp {
     color: var(--color-text-muted);
     min-width: 80px;
     flex-shrink: 0;
   }

   .log-level {
     min-width: 60px;
     flex-shrink: 0;
     font-weight: var(--font-weight-medium);
   }

   .log-entry.log-error .log-level {
     color: var(--color-error);
   }

   .log-entry.log-warning .log-level {
     color: var(--color-warning);
   }

   .log-entry.log-info .log-level {
     color: var(--color-info);
   }

   .log-entry.log-success .log-level {
     color: var(--color-success);
   }

   .log-message {
     flex: 1;
     color: var(--color-text);
   }

   .log-controls {
     display: flex;
     gap: var(--spacing-md);
     justify-content: flex-end;
   }

   /* Footer */
   .app-footer {
     background: var(--color-surface);
     border-top: 1px solid var(--color-border);
     margin-top: auto;
   }

   .footer-content {
     display: flex;
     align-items: center;
     justify-content: space-between;
     padding: var(--spacing-lg);
     max-width: 1200px;
     margin: 0 auto;
   }

   .footer-info p {
     font-size: var(--font-size-sm);
     color: var(--color-text-muted);
   }

   .footer-links {
     display: flex;
     gap: var(--spacing-lg);
   }

   .footer-link {
     background: none;
     border: none;
     font-size: var(--font-size-sm);
     color: var(--color-text-secondary);
     cursor: pointer;
     text-decoration: none;
     transition: color var(--transition-fast);
   }

   .footer-link:hover {
     color: var(--color-primary);
   }

   /* Utility Classes */
   .hidden {
     display: none !important;
   }

   .sr-only {
     position: absolute;
     width: 1px;
     height: 1px;
     padding: 0;
     margin: -1px;
     overflow: hidden;
     clip: rect(0, 0, 0, 0);
     white-space: nowrap;
     border: 0;
   }

   .truncate {
     overflow: hidden;
     text-overflow: ellipsis;
     white-space: nowrap;
   }

   .text-center {
     text-align: center;
   }

   .text-right {
     text-align: right;
   }

   .font-mono {
     font-family: var(--font-family-mono);
   }

   /* Responsive Utilities */
   @media (max-width: 768px) {
     .metrics-grid {
       grid-template-columns: 1fr;
     }

     .monitor-charts {
       grid-template-columns: 1fr;
     }

     .viz-controls {
       flex-direction: column;
       align-items: stretch;
     }

     .control-group {
       justify-content: space-between;
     }

     .footer-content {
       flex-direction: column;
       gap: var(--spacing-md);
       text-align: center;
     }

     .toast-container {
       left: var(--spacing-sm);
       right: var(--spacing-sm);
       max-width: none;
     }
   }
   ```

## Expected Outputs
- Complete responsive CSS design system with design tokens
- Dark/light theme support with system preference detection
- Accessible components with proper focus management
- Smooth animations and transitions with reduced motion support
- Mobile-first responsive design for all screen sizes
- Print-friendly styles and high contrast mode support

## Validation
1. Design works seamlessly across desktop, tablet, and mobile devices
2. Dark and light themes function correctly with smooth transitions
3. All interactive elements meet WCAG accessibility guidelines
4. Color contrast ratios exceed 4.5:1 for normal text, 3:1 for large text
5. CSS validates with no errors and follows BEM/utility naming conventions

## Next Steps
- Integrate CSS with JavaScript components and interactions
- Test across different browsers and devices for compatibility
- Optimize performance and loading times