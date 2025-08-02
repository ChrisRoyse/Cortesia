# Micro-Phase 9.21: HTML Structure Implementation

## Objective
Create a semantic, accessible HTML structure for the CortexKG web interface with proper components, navigation, and interactive elements for knowledge graph visualization and interaction.

## Prerequisites
- Completed micro-phase 9.20 (Error handling)
- TypeScript project structure configured
- CSS framework decisions made
- Accessibility requirements defined

## Task Description
Design and implement a comprehensive HTML structure that provides a clean, accessible interface for CortexKG operations including concept allocation, querying, visualization, and system monitoring with proper semantic markup and ARIA support.

## Specific Actions

1. **Create main HTML template**:
   ```html
   <!-- public/index.html -->
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <meta name="description" content="CortexKG - Advanced Knowledge Graph Interface">
     <meta name="theme-color" content="#1a365d">
     <title>CortexKG - Knowledge Graph Interface</title>
     
     <!-- Preload critical resources -->
     <link rel="preload" href="./wasm/cortexkg_wasm.wasm" as="fetch" type="application/wasm" crossorigin>
     <link rel="preload" href="./cortexkg.bundle.js" as="script">
     
     <!-- PWA manifest -->
     <link rel="manifest" href="./manifest.json">
     
     <!-- Favicon -->
     <link rel="icon" type="image/svg+xml" href="./favicon.svg">
     <link rel="icon" type="image/png" href="./favicon.png">
     
     <!-- CSS -->
     <link rel="stylesheet" href="./styles/main.css">
     
     <!-- WASM headers for security -->
     <meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
     <meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
   </head>
   <body>
     <!-- Loading screen -->
     <div id="loading-screen" class="loading-screen" aria-hidden="true">
       <div class="loading-spinner" role="status" aria-label="Loading CortexKG">
         <div class="spinner-circle"></div>
         <div class="loading-text">
           <h2>Initializing CortexKG</h2>
           <p id="loading-status">Loading WebAssembly module...</p>
           <div class="progress-bar">
             <div id="loading-progress" class="progress-fill" style="width: 0%"></div>
           </div>
         </div>
       </div>
     </div>

     <!-- Main application -->
     <div id="app" class="app-container" aria-hidden="true">
       <!-- Header -->
       <header class="app-header" role="banner">
         <div class="header-content">
           <div class="logo-section">
             <svg class="logo-icon" viewBox="0 0 24 24" aria-hidden="true">
               <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 9.96 5.16-.22 9-4.41 9-9.96V7l-10-5z"/>
               <circle cx="12" cy="12" r="3"/>
               <path d="M12 1L3 5v6c0 5.55 3.84 9.74 9 9.96 5.16-.22 9-4.41 9-9.96V5l-9-4z"/>
             </svg>
             <h1 class="app-title">CortexKG</h1>
           </div>
           
           <nav class="main-navigation" role="navigation" aria-label="Main navigation">
             <ul class="nav-list">
               <li><button class="nav-button active" data-tab="allocate" aria-pressed="true">Allocate</button></li>
               <li><button class="nav-button" data-tab="query" aria-pressed="false">Query</button></li>
               <li><button class="nav-button" data-tab="visualize" aria-pressed="false">Visualize</button></li>
               <li><button class="nav-button" data-tab="monitor" aria-pressed="false">Monitor</button></li>
             </ul>
           </nav>
           
           <div class="header-actions">
             <button id="settings-button" class="icon-button" aria-label="Settings" title="Settings">
               <svg viewBox="0 0 24 24" aria-hidden="true">
                 <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L19 6.8C18.9 6.5 18.8 6.2 18.7 5.9L20.1 4.5L18.5 2.9L17.1 4.3C16.8 4.2 16.5 4.1 16.2 4L16 2H14L13.8 4C13.5 4.1 13.2 4.2 12.9 4.3L11.5 2.9L9.9 4.5L11.3 5.9C11.2 6.2 11.1 6.5 11 6.8L9 7V9L11 9.2C11.1 9.5 11.2 9.8 11.3 10.1L9.9 11.5L11.5 13.1L12.9 11.7C13.2 11.8 13.5 11.9 13.8 12L14 14H16L16.2 12C16.5 11.9 16.8 11.8 17.1 11.7L18.5 13.1L20.1 11.5L18.7 10.1C18.8 9.8 18.9 9.5 19 9.2L21 9ZM12 15C10.3 15 9 13.7 9 12C9 10.3 10.3 9 12 9C13.7 9 15 10.3 15 12C15 13.7 13.7 15 12 15Z"/>
               </svg>
             </button>
             
             <div class="status-indicator">
               <div id="connection-status" class="status-dot status-connecting" 
                    aria-label="Connection status" role="status">
               </div>
               <span class="status-text" id="status-text">Initializing</span>
             </div>
           </div>
         </div>
       </header>

       <!-- Main content area -->
       <main class="main-content" role="main">
         <!-- Allocation Tab -->
         <section id="tab-allocate" class="tab-content active" role="tabpanel" 
                  aria-labelledby="nav-allocate" aria-hidden="false">
           <div class="section-header">
             <h2>Concept Allocation</h2>
             <p>Add new concepts to the knowledge graph</p>
           </div>
           
           <div class="allocation-interface">
             <form id="allocation-form" class="input-form" novalidate>
               <div class="form-group">
                 <label for="concept-input" class="form-label">
                   Concept Content
                   <span class="required" aria-label="required">*</span>
                 </label>
                 <textarea 
                   id="concept-input" 
                   class="form-textarea"
                   placeholder="Enter the concept or fact to allocate..."
                   required
                   aria-describedby="concept-help concept-error"
                   rows="4"
                   maxlength="10000"
                 ></textarea>
                 <div id="concept-help" class="form-help">
                   Enter a clear, descriptive concept or fact. Maximum 10,000 characters.
                 </div>
                 <div id="concept-error" class="form-error" role="alert" aria-live="polite"></div>
                 <div class="character-count">
                   <span id="char-count">0</span> / 10,000
                 </div>
               </div>
               
               <div class="form-actions">
                 <button type="submit" id="allocate-button" class="primary-button">
                   <span class="button-text">Allocate Concept</span>
                   <div class="button-spinner" aria-hidden="true"></div>
                 </button>
                 <button type="button" id="clear-button" class="secondary-button">Clear</button>
               </div>
             </form>
             
             <div id="allocation-results" class="results-section" aria-live="polite">
               <!-- Results will be populated here -->
             </div>
           </div>
         </section>

         <!-- Query Tab -->
         <section id="tab-query" class="tab-content" role="tabpanel" 
                  aria-labelledby="nav-query" aria-hidden="true">
           <div class="section-header">
             <h2>Knowledge Query</h2>
             <p>Search and explore the knowledge graph</p>
           </div>
           
           <div class="query-interface">
             <form id="query-form" class="input-form" novalidate>
               <div class="form-group">
                 <label for="query-input" class="form-label">
                   Query Text
                   <span class="required" aria-label="required">*</span>
                 </label>
                 <input 
                   type="text" 
                   id="query-input" 
                   class="form-input"
                   placeholder="Enter your query..."
                   required
                   aria-describedby="query-help query-error"
                   maxlength="1000"
                 />
                 <div id="query-help" class="form-help">
                   Enter keywords or phrases to search the knowledge graph.
                 </div>
                 <div id="query-error" class="form-error" role="alert" aria-live="polite"></div>
               </div>
               
               <div class="query-options">
                 <div class="form-group inline">
                   <label for="max-results" class="form-label">Max Results</label>
                   <select id="max-results" class="form-select">
                     <option value="5">5</option>
                     <option value="10" selected>10</option>
                     <option value="25">25</option>
                     <option value="50">50</option>
                   </select>
                 </div>
                 
                 <div class="form-group inline">
                   <label for="threshold" class="form-label">Relevance Threshold</label>
                   <input 
                     type="range" 
                     id="threshold" 
                     class="form-range"
                     min="0" 
                     max="1" 
                     step="0.1" 
                     value="0.5"
                     aria-describedby="threshold-value"
                   />
                   <span id="threshold-value" class="range-value">0.5</span>
                 </div>
               </div>
               
               <div class="form-actions">
                 <button type="submit" id="query-button" class="primary-button">
                   <span class="button-text">Search</span>
                   <div class="button-spinner" aria-hidden="true"></div>
                 </button>
                 <button type="button" id="clear-query-button" class="secondary-button">Clear</button>
               </div>
             </form>
             
             <div id="query-results" class="results-section" aria-live="polite">
               <!-- Query results will be populated here -->
             </div>
           </div>
         </section>

         <!-- Visualization Tab -->
         <section id="tab-visualize" class="tab-content" role="tabpanel" 
                  aria-labelledby="nav-visualize" aria-hidden="true">
           <div class="section-header">
             <h2>Cortical Visualization</h2>
             <p>Interactive visualization of the cortical column network</p>
           </div>
           
           <div class="visualization-interface">
             <div class="viz-controls">
               <div class="control-group">
                 <button id="play-pause-button" class="icon-button" aria-label="Play/Pause animation">
                   <svg class="play-icon" viewBox="0 0 24 24" aria-hidden="true">
                     <polygon points="5,3 19,12 5,21"></polygon>
                   </svg>
                   <svg class="pause-icon hidden" viewBox="0 0 24 24" aria-hidden="true">
                     <rect x="6" y="4" width="4" height="16"></rect>
                     <rect x="14" y="4" width="4" height="16"></rect>
                   </svg>
                 </button>
                 
                 <button id="reset-view-button" class="icon-button" aria-label="Reset view">
                   <svg viewBox="0 0 24 24" aria-hidden="true">
                     <path d="M12 2v10l4-4-4-4z"/>
                     <path d="M12 14v8l-4-4 4-4z"/>
                   </svg>
                 </button>
               </div>
               
               <div class="control-group">
                 <label for="zoom-control" class="control-label">Zoom</label>
                 <input 
                   type="range" 
                   id="zoom-control" 
                   class="form-range"
                   min="0.1" 
                   max="3" 
                   step="0.1" 
                   value="1"
                   aria-describedby="zoom-value"
                 />
                 <span id="zoom-value" class="range-value">1.0x</span>
               </div>
               
               <div class="control-group">
                 <label for="animation-speed" class="control-label">Animation Speed</label>
                 <input 
                   type="range" 
                   id="animation-speed" 
                   class="form-range"
                   min="0.1" 
                   max="2" 
                   step="0.1" 
                   value="1"
                   aria-describedby="speed-value"
                 />
                 <span id="speed-value" class="range-value">1.0x</span>
               </div>
             </div>
             
             <div class="visualization-container">
               <canvas 
                 id="cortical-canvas" 
                 class="cortical-canvas"
                 role="img"
                 aria-label="Cortical column network visualization"
                 tabindex="0"
               ></canvas>
               
               <div class="viz-overlay">
                 <div id="column-info" class="info-panel" aria-live="polite">
                   <h3>Column Information</h3>
                   <p>Hover over a column to see details</p>
                 </div>
               </div>
             </div>
           </div>
         </section>

         <!-- Monitor Tab -->
         <section id="tab-monitor" class="tab-content" role="tabpanel" 
                  aria-labelledby="nav-monitor" aria-hidden="true">
           <div class="section-header">
             <h2>System Monitor</h2>
             <p>Performance metrics and system status</p>
           </div>
           
           <div class="monitor-interface">
             <div class="metrics-grid">
               <div class="metric-card">
                 <h3 class="metric-title">Allocations</h3>
                 <div class="metric-value" id="total-allocations">0</div>
                 <div class="metric-subtitle">Total concepts allocated</div>
               </div>
               
               <div class="metric-card">
                 <h3 class="metric-title">Avg. Allocation Time</h3>
                 <div class="metric-value" id="avg-allocation-time">0ms</div>
                 <div class="metric-subtitle">Average processing time</div>
               </div>
               
               <div class="metric-card">
                 <h3 class="metric-title">Memory Usage</h3>
                 <div class="metric-value" id="memory-usage">0MB</div>
                 <div class="metric-subtitle">Current WASM memory</div>
               </div>
               
               <div class="metric-card">
                 <h3 class="metric-title">Cache Hit Rate</h3>
                 <div class="metric-value" id="cache-hit-rate">0%</div>
                 <div class="metric-subtitle">Query cache efficiency</div>
               </div>
             </div>
             
             <div class="monitor-charts">
               <div class="chart-container">
                 <h3>Allocation Timeline</h3>
                 <canvas id="allocation-chart" class="performance-chart"></canvas>
               </div>
               
               <div class="chart-container">
                 <h3>Memory Usage</h3>
                 <canvas id="memory-chart" class="performance-chart"></canvas>
               </div>
             </div>
             
             <div class="system-logs">
               <h3>System Logs</h3>
               <div id="log-container" class="log-container" role="log" aria-live="polite">
                 <!-- Log entries will be populated here -->
               </div>
               <div class="log-controls">
                 <button id="clear-logs-button" class="secondary-button">Clear Logs</button>
                 <button id="export-logs-button" class="secondary-button">Export Logs</button>
               </div>
             </div>
           </div>
         </section>
       </main>

       <!-- Footer -->
       <footer class="app-footer" role="contentinfo">
         <div class="footer-content">
           <div class="footer-info">
             <p>&copy; 2024 CortexKG. Advanced Knowledge Graph System.</p>
           </div>
           <div class="footer-links">
             <button id="about-button" class="footer-link">About</button>
             <button id="help-button" class="footer-link">Help</button>
             <button id="privacy-button" class="footer-link">Privacy</button>
           </div>
         </div>
       </footer>
     </div>

     <!-- Settings Modal -->
     <div id="settings-modal" class="modal" role="dialog" aria-labelledby="settings-title" aria-hidden="true">
       <div class="modal-backdrop"></div>
       <div class="modal-content">
         <header class="modal-header">
           <h2 id="settings-title">Settings</h2>
           <button class="modal-close" aria-label="Close settings">
             <svg viewBox="0 0 24 24" aria-hidden="true">
               <path d="M18 6L6 18M6 6l12 12"/>
             </svg>
           </button>
         </header>
         
         <div class="modal-body">
           <div class="settings-section">
             <h3>Performance</h3>
             <div class="setting-item">
               <label class="setting-label">
                 <input type="checkbox" id="enable-simd" checked>
                 Enable SIMD Acceleration
               </label>
               <p class="setting-description">Use SIMD instructions for faster processing (requires browser support)</p>
             </div>
             
             <div class="setting-item">
               <label for="cache-size" class="setting-label">Cache Size (MB)</label>
               <input type="number" id="cache-size" min="16" max="512" value="64" class="setting-input">
               <p class="setting-description">Amount of memory to use for caching</p>
             </div>
           </div>
           
           <div class="settings-section">
             <h3>Interface</h3>
             <div class="setting-item">
               <label class="setting-label">
                 <input type="checkbox" id="enable-animations" checked>
                 Enable Animations
               </label>
               <p class="setting-description">Show animated transitions and effects</p>
             </div>
             
             <div class="setting-item">
               <label for="theme-select" class="setting-label">Theme</label>
               <select id="theme-select" class="setting-select">
                 <option value="auto">Auto (System)</option>
                 <option value="light">Light</option>
                 <option value="dark">Dark</option>
               </select>
             </div>
           </div>
         </div>
         
         <footer class="modal-footer">
           <button id="settings-save" class="primary-button">Save Settings</button>
           <button id="settings-cancel" class="secondary-button">Cancel</button>
         </footer>
       </div>
     </div>

     <!-- Error Toast Container -->
     <div id="toast-container" class="toast-container" aria-live="assertive" aria-atomic="true">
       <!-- Toast notifications will be populated here -->
     </div>

     <!-- Skip link for accessibility -->
     <a href="#main-content" class="skip-link">Skip to main content</a>

     <!-- Scripts -->
     <script src="./cortexkg.bundle.js"></script>
     <script>
       // Initialize the application
       window.addEventListener('DOMContentLoaded', () => {
         if (window.CortexKG) {
           window.cortexApp = new window.CortexKG.CortexKGWeb({
             wasmPath: './wasm/cortexkg_wasm.wasm',
             autoInitialize: true,
             enablePersistence: true,
             databaseName: 'cortexkg_local',
             debugMode: false
           });
         }
       });
     </script>
   </body>
   </html>
   ```

2. **Create PWA manifest**:
   ```json
   <!-- public/manifest.json -->
   {
     "name": "CortexKG - Knowledge Graph Interface",
     "short_name": "CortexKG",
     "description": "Advanced knowledge graph system with cortical column architecture",
     "start_url": "/",
     "display": "standalone",
     "background_color": "#1a202c",
     "theme_color": "#1a365d",
     "orientation": "portrait-primary",
     "categories": ["productivity", "education", "science"],
     "lang": "en",
     "icons": [
       {
         "src": "./icons/icon-72x72.png",
         "sizes": "72x72",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-96x96.png",
         "sizes": "96x96",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-128x128.png",
         "sizes": "128x128",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-144x144.png",
         "sizes": "144x144",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-152x152.png",
         "sizes": "152x152",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-192x192.png",
         "sizes": "192x192",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-384x384.png",
         "sizes": "384x384",
         "type": "image/png"
       },
       {
         "src": "./icons/icon-512x512.png",
         "sizes": "512x512",
         "type": "image/png"
       }
     ],
     "screenshots": [
       {
         "src": "./screenshots/desktop.png",
         "sizes": "1280x720",
         "type": "image/png",
         "form_factor": "wide"
       },
       {
         "src": "./screenshots/mobile.png",
         "sizes": "375x667",
         "type": "image/png",
         "form_factor": "narrow"
       }
     ]
   }
   ```

3. **Create HTML component templates**:
   ```typescript
   // src/ui/templates/HTMLTemplates.ts
   export class HTMLTemplates {
     static allocationResult(result: AllocationResult): string {
       return `
         <div class="result-card success" role="article">
           <div class="result-header">
             <h3>Allocation Successful</h3>
             <span class="result-timestamp">${new Date().toLocaleTimeString()}</span>
           </div>
           <div class="result-body">
             <div class="result-metric">
               <span class="metric-label">Column ID:</span>
               <span class="metric-value">${result.column_id}</span>
             </div>
             <div class="result-metric">
               <span class="metric-label">Confidence:</span>
               <span class="metric-value">${(result.confidence * 100).toFixed(1)}%</span>
             </div>
             <div class="result-metric">
               <span class="metric-label">Processing Time:</span>
               <span class="metric-value">${result.processing_time_ms.toFixed(2)}ms</span>
             </div>
           </div>
         </div>
       `;
     }

     static queryResult(result: QueryResult, index: number): string {
       return `
         <div class="query-result-item" role="article" tabindex="0">
           <div class="result-rank">${index + 1}</div>
           <div class="result-content">
             <div class="result-text">${this.escapeHtml(result.content)}</div>
             <div class="result-meta">
               <span class="relevance-score">
                 Relevance: ${(result.relevance_score * 100).toFixed(1)}%
               </span>
               <span class="concept-id">ID: ${result.concept_id}</span>
             </div>
           </div>
           <div class="result-actions">
             <button class="action-button" data-action="view-details" data-concept-id="${result.concept_id}">
               Details
             </button>
           </div>
         </div>
       `;
     }

     static errorToast(error: CortexKGError): string {
       const severityClass = `toast-${error.severity}`;
       return `
         <div class="toast ${severityClass}" role="alert" aria-live="assertive">
           <div class="toast-icon">
             ${this.getErrorIcon(error.severity)}
           </div>
           <div class="toast-content">
             <div class="toast-title">${error.category.replace('_', ' ').toUpperCase()}</div>
             <div class="toast-message">${this.escapeHtml(error.userFriendlyMessage)}</div>
           </div>
           <button class="toast-close" aria-label="Dismiss notification">
             <svg viewBox="0 0 24 24" aria-hidden="true">
               <path d="M18 6L6 18M6 6l12 12"/>
             </svg>
           </button>
         </div>
       `;
     }

     static logEntry(level: string, message: string, timestamp: Date): string {
       const levelClass = `log-${level.toLowerCase()}`;
       return `
         <div class="log-entry ${levelClass}">
           <span class="log-timestamp">${timestamp.toLocaleTimeString()}</span>
           <span class="log-level">${level.toUpperCase()}</span>
           <span class="log-message">${this.escapeHtml(message)}</span>
         </div>
       `;
     }

     static metricCard(title: string, value: string, subtitle: string): string {
       return `
         <div class="metric-card">
           <h3 class="metric-title">${this.escapeHtml(title)}</h3>
           <div class="metric-value">${this.escapeHtml(value)}</div>
           <div class="metric-subtitle">${this.escapeHtml(subtitle)}</div>
         </div>
       `;
     }

     private static escapeHtml(text: string): string {
       const div = document.createElement('div');
       div.textContent = text;
       return div.innerHTML;
     }

     private static getErrorIcon(severity: string): string {
       switch (severity) {
         case 'low':
           return '<svg viewBox="0 0 24 24"><path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 9.96V2z"/></svg>';
         case 'medium':
           return '<svg viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>';
         case 'high':
         case 'critical':
           return '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
         default:
           return '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
       }
     }
   }
   ```

## Expected Outputs
- Complete semantic HTML structure with proper ARIA attributes
- Accessible navigation and form controls
- PWA manifest for installable web app
- Reusable HTML template system
- Proper loading states and error handling UI
- Responsive layout structure ready for CSS styling

## Validation
1. HTML validates against W3C standards with no errors
2. All interactive elements have proper ARIA attributes and labels
3. Keyboard navigation works throughout the interface
4. Screen reader compatibility verified with NVDA/JAWS
5. PWA manifest generates valid installable app

## Next Steps
- Implement responsive CSS design (micro-phase 9.22)
- Integrate with JavaScript API wrapper for functionality