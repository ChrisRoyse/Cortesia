# Micro-Phase 9.24: Query Interface Component

## Objective
Create a comprehensive query interface with autocomplete, filtering, and search capabilities for interacting with the neuromorphic knowledge graph.

## Prerequisites
- Completed micro-phase 9.23 (Cortical Visualizer)
- WASM module with query methods available
- Understanding of TypeScript/JavaScript event handling

## Task Description
Implement an intelligent query interface that provides real-time suggestions, filters results by concept type, and maintains query history for enhanced user experience.

## Specific Actions

1. **Create QueryInterface class structure**:
   ```typescript
   // src/ui/QueryInterface.ts
   import { CortexKGWasm } from 'cortexkg-wasm';
   
   export interface QueryConfig {
     container: HTMLElement;
     placeholder?: string;
     maxSuggestions?: number;
     enableHistory?: boolean;
     enableFilters?: boolean;
     onQuery?: (query: string, filters: QueryFilters) => void;
     onClear?: () => void;
   }
   
   export interface QueryFilters {
     conceptTypes: string[];
     minActivation: number;
     maxResults: number;
     sortBy: 'relevance' | 'activation' | 'created' | 'alphabetical';
     timeRange?: [Date, Date];
   }
   
   export interface QuerySuggestion {
     text: string;
     type: 'concept' | 'relationship' | 'semantic';
     score: number;
     metadata?: any;
   }
   
   export class QueryInterface {
     private container: HTMLElement;
     private wasmModule: CortexKGWasm;
     private config: Required<QueryConfig>;
     private inputElement: HTMLInputElement;
     private suggestionsContainer: HTMLElement;
     private filtersContainer: HTMLElement;
     private historyContainer: HTMLElement;
     private currentSuggestions: QuerySuggestion[] = [];
     private queryHistory: string[] = [];
     private activeFilters: QueryFilters;
     private debounceTimer: number | null = null;
     
     constructor(wasmModule: CortexKGWasm, config: QueryConfig) {
       this.wasmModule = wasmModule;
       this.container = config.container;
       this.config = {
         container: config.container,
         placeholder: config.placeholder ?? 'Search concepts, relationships...',
         maxSuggestions: config.maxSuggestions ?? 10,
         enableHistory: config.enableHistory ?? true,
         enableFilters: config.enableFilters ?? true,
         onQuery: config.onQuery ?? (() => {}),
         onClear: config.onClear ?? (() => {})
       };
       
       this.activeFilters = this.getDefaultFilters();
       this.createInterface();
       this.setupEventHandlers();
       this.loadQueryHistory();
     }
   }
   ```

2. **Implement interface creation and DOM structure**:
   ```typescript
   private createInterface(): void {
     this.container.innerHTML = `
       <div class="query-interface">
         <div class="query-input-section">
           <div class="input-wrapper">
             <input type="text" class="query-input" placeholder="${this.config.placeholder}">
             <button class="clear-button" title="Clear">×</button>
             <button class="search-button" title="Search">🔍</button>
           </div>
           <div class="suggestions-container"></div>
         </div>
         
         ${this.config.enableFilters ? `
         <div class="filters-section">
           <div class="filters-toggle">
             <button class="toggle-filters">Filters</button>
             <span class="active-filters-count">0</span>
           </div>
           <div class="filters-container collapsed">
             <div class="filter-group">
               <label>Concept Types:</label>
               <div class="concept-types-filter">
                 <label><input type="checkbox" value="entity" checked> Entities</label>
                 <label><input type="checkbox" value="relationship" checked> Relationships</label>
                 <label><input type="checkbox" value="attribute" checked> Attributes</label>
                 <label><input type="checkbox" value="semantic" checked> Semantic</label>
               </div>
             </div>
             
             <div class="filter-group">
               <label>Minimum Activation:</label>
               <input type="range" class="activation-slider" min="0" max="1" step="0.1" value="0">
               <span class="activation-value">0.0</span>
             </div>
             
             <div class="filter-group">
               <label>Max Results:</label>
               <select class="max-results-select">
                 <option value="10">10</option>
                 <option value="25" selected>25</option>
                 <option value="50">50</option>
                 <option value="100">100</option>
               </select>
             </div>
             
             <div class="filter-group">
               <label>Sort By:</label>
               <select class="sort-select">
                 <option value="relevance" selected>Relevance</option>
                 <option value="activation">Activation</option>
                 <option value="created">Created Date</option>
                 <option value="alphabetical">Alphabetical</option>
               </select>
             </div>
           </div>
         </div>
         ` : ''}
         
         ${this.config.enableHistory ? `
         <div class="history-section">
           <div class="history-toggle">
             <button class="toggle-history">Recent Queries</button>
           </div>
           <div class="history-container collapsed"></div>
         </div>
         ` : ''}
       </div>
     `;
     
     // Get references to key elements
     this.inputElement = this.container.querySelector('.query-input') as HTMLInputElement;
     this.suggestionsContainer = this.container.querySelector('.suggestions-container') as HTMLElement;
     this.filtersContainer = this.container.querySelector('.filters-container') as HTMLElement;
     this.historyContainer = this.container.querySelector('.history-container') as HTMLElement;
   }
   ```

3. **Implement real-time autocomplete and suggestions**:
   ```typescript
   private setupEventHandlers(): void {
     // Input handling with debounce
     this.inputElement.addEventListener('input', (e) => {
       const query = (e.target as HTMLInputElement).value;
       
       if (this.debounceTimer) {
         clearTimeout(this.debounceTimer);
       }
       
       this.debounceTimer = window.setTimeout(() => {
         this.handleQueryInput(query);
       }, 200);
     });
     
     // Enter key to execute query
     this.inputElement.addEventListener('keydown', (e) => {
       if (e.key === 'Enter') {
         e.preventDefault();
         this.executeQuery();
       } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
         e.preventDefault();
         this.navigateSuggestions(e.key === 'ArrowDown' ? 1 : -1);
       } else if (e.key === 'Escape') {
         this.hideSuggestions();
       }
     });
     
     // Search button
     this.container.querySelector('.search-button')?.addEventListener('click', () => {
       this.executeQuery();
     });
     
     // Clear button
     this.container.querySelector('.clear-button')?.addEventListener('click', () => {
       this.clearQuery();
     });
     
     // Filter controls
     if (this.config.enableFilters) {
       this.setupFilterHandlers();
     }
     
     // History controls
     if (this.config.enableHistory) {
       this.setupHistoryHandlers();
     }
   }
   
   private async handleQueryInput(query: string): Promise<void> {
     if (query.length < 2) {
       this.hideSuggestions();
       return;
     }
     
     try {
       // Get suggestions from WASM module
       const suggestions = await this.getSuggestions(query);
       this.displaySuggestions(suggestions);
     } catch (error) {
       console.error('Error getting suggestions:', error);
       this.hideSuggestions();
     }
   }
   
   private async getSuggestions(query: string): Promise<QuerySuggestion[]> {
     // Use semantic search to find related concepts
     const semanticResults = this.wasmModule.semantic_search(query, 5);
     const conceptResults = this.wasmModule.find_concepts_by_prefix(query, 5);
     
     const suggestions: QuerySuggestion[] = [];
     
     // Add semantic suggestions
     semanticResults.forEach((result: any) => {
       suggestions.push({
         text: result.concept,
         type: 'semantic',
         score: result.similarity,
         metadata: { conceptId: result.id }
       });
     });
     
     // Add concept prefix matches
     conceptResults.forEach((concept: any) => {
       if (!suggestions.find(s => s.text === concept.name)) {
         suggestions.push({
           text: concept.name,
           type: 'concept',
           score: 0.8,
           metadata: { conceptId: concept.id }
         });
       }
     });
     
     // Sort by score and limit
     return suggestions
       .sort((a, b) => b.score - a.score)
       .slice(0, this.config.maxSuggestions);
   }
   ```

4. **Implement suggestion display and interaction**:
   ```typescript
   private displaySuggestions(suggestions: QuerySuggestion[]): void {
     this.currentSuggestions = suggestions;
     
     if (suggestions.length === 0) {
       this.hideSuggestions();
       return;
     }
     
     const html = suggestions.map((suggestion, index) => `
       <div class="suggestion-item ${index === 0 ? 'selected' : ''}" 
            data-index="${index}" 
            data-text="${suggestion.text}">
         <div class="suggestion-text">${this.highlightMatch(suggestion.text)}</div>
         <div class="suggestion-meta">
           <span class="suggestion-type">${suggestion.type}</span>
           <span class="suggestion-score">${(suggestion.score * 100).toFixed(0)}%</span>
         </div>
       </div>
     `).join('');
     
     this.suggestionsContainer.innerHTML = html;
     this.suggestionsContainer.classList.add('visible');
     
     // Add click handlers
     this.suggestionsContainer.querySelectorAll('.suggestion-item').forEach(item => {
       item.addEventListener('click', () => {
         const text = item.getAttribute('data-text')!;
         this.selectSuggestion(text);
       });
     });
   }
   
   private highlightMatch(text: string): string {
     const query = this.inputElement.value.toLowerCase();
     const index = text.toLowerCase().indexOf(query);
     
     if (index === -1) return text;
     
     return text.substring(0, index) + 
            `<mark>${text.substring(index, index + query.length)}</mark>` + 
            text.substring(index + query.length);
   }
   
   private navigateSuggestions(direction: number): void {
     const items = this.suggestionsContainer.querySelectorAll('.suggestion-item');
     const current = this.suggestionsContainer.querySelector('.suggestion-item.selected');
     
     if (!current) return;
     
     const currentIndex = Array.from(items).indexOf(current);
     const newIndex = Math.max(0, Math.min(items.length - 1, currentIndex + direction));
     
     current.classList.remove('selected');
     items[newIndex].classList.add('selected');
   }
   
   private selectSuggestion(text: string): void {
     this.inputElement.value = text;
     this.hideSuggestions();
     this.executeQuery();
   }
   
   private hideSuggestions(): void {
     this.suggestionsContainer.classList.remove('visible');
     this.currentSuggestions = [];
   }
   ```

5. **Implement filter management and query execution**:
   ```typescript
   private setupFilterHandlers(): void {
     // Toggle filters visibility
     this.container.querySelector('.toggle-filters')?.addEventListener('click', () => {
       this.filtersContainer.classList.toggle('collapsed');
     });
     
     // Concept type filters
     this.container.querySelectorAll('.concept-types-filter input').forEach(checkbox => {
       checkbox.addEventListener('change', () => {
         this.updateFilters();
       });
     });
     
     // Activation slider
     const slider = this.container.querySelector('.activation-slider') as HTMLInputElement;
     const valueDisplay = this.container.querySelector('.activation-value') as HTMLElement;
     
     slider?.addEventListener('input', () => {
       valueDisplay.textContent = parseFloat(slider.value).toFixed(1);
       this.updateFilters();
     });
     
     // Max results and sort options
     this.container.querySelector('.max-results-select')?.addEventListener('change', () => {
       this.updateFilters();
     });
     
     this.container.querySelector('.sort-select')?.addEventListener('change', () => {
       this.updateFilters();
     });
   }
   
   private updateFilters(): void {
     const conceptTypes: string[] = [];
     this.container.querySelectorAll('.concept-types-filter input:checked').forEach(checkbox => {
       conceptTypes.push((checkbox as HTMLInputElement).value);
     });
     
     const minActivation = parseFloat((this.container.querySelector('.activation-slider') as HTMLInputElement).value);
     const maxResults = parseInt((this.container.querySelector('.max-results-select') as HTMLSelectElement).value);
     const sortBy = (this.container.querySelector('.sort-select') as HTMLSelectElement).value as QueryFilters['sortBy'];
     
     this.activeFilters = {
       conceptTypes,
       minActivation,
       maxResults,
       sortBy
     };
     
     // Update active filters count
     const activeCount = this.getActiveFilterCount();
     const countDisplay = this.container.querySelector('.active-filters-count') as HTMLElement;
     countDisplay.textContent = activeCount.toString();
     countDisplay.style.display = activeCount > 0 ? 'inline' : 'none';
   }
   
   private executeQuery(): void {
     const query = this.inputElement.value.trim();
     
     if (!query) return;
     
     // Add to history
     if (this.config.enableHistory) {
       this.addToHistory(query);
     }
     
     // Hide suggestions
     this.hideSuggestions();
     
     // Execute query with current filters
     this.config.onQuery(query, this.activeFilters);
   }
   
   private getDefaultFilters(): QueryFilters {
     return {
       conceptTypes: ['entity', 'relationship', 'attribute', 'semantic'],
       minActivation: 0.0,
       maxResults: 25,
       sortBy: 'relevance'
     };
   }
   
   private getActiveFilterCount(): number {
     const defaults = this.getDefaultFilters();
     let count = 0;
     
     if (this.activeFilters.conceptTypes.length !== defaults.conceptTypes.length) count++;
     if (this.activeFilters.minActivation !== defaults.minActivation) count++;
     if (this.activeFilters.maxResults !== defaults.maxResults) count++;
     if (this.activeFilters.sortBy !== defaults.sortBy) count++;
     
     return count;
   }
   
   public clearQuery(): void {
     this.inputElement.value = '';
     this.hideSuggestions();
     this.config.onClear();
   }
   
   public setQuery(query: string): void {
     this.inputElement.value = query;
     this.inputElement.focus();
   }
   
   public getFilters(): QueryFilters {
     return { ...this.activeFilters };
   }
   ```

6. **Implement query history management**:
   ```typescript
   private setupHistoryHandlers(): void {
     this.container.querySelector('.toggle-history')?.addEventListener('click', () => {
       this.historyContainer.classList.toggle('collapsed');
       this.updateHistoryDisplay();
     });
   }
   
   private addToHistory(query: string): void {
     // Remove duplicates and add to front
     this.queryHistory = [query, ...this.queryHistory.filter(q => q !== query)];
     
     // Limit history size
     if (this.queryHistory.length > 20) {
       this.queryHistory = this.queryHistory.slice(0, 20);
     }
     
     // Save to localStorage
     this.saveQueryHistory();
   }
   
   private updateHistoryDisplay(): void {
     if (!this.historyContainer || this.historyContainer.classList.contains('collapsed')) {
       return;
     }
     
     const html = this.queryHistory.map(query => `
       <div class="history-item" data-query="${query}">
         <span class="history-text">${query}</span>
         <button class="history-remove" title="Remove">×</button>
       </div>
     `).join('');
     
     this.historyContainer.innerHTML = html || '<div class="no-history">No recent queries</div>';
     
     // Add event handlers
     this.historyContainer.querySelectorAll('.history-item').forEach(item => {
       const queryText = item.getAttribute('data-query')!;
       
       item.querySelector('.history-text')?.addEventListener('click', () => {
         this.setQuery(queryText);
       });
       
       item.querySelector('.history-remove')?.addEventListener('click', (e) => {
         e.stopPropagation();
         this.removeFromHistory(queryText);
       });
     });
   }
   
   private removeFromHistory(query: string): void {
     this.queryHistory = this.queryHistory.filter(q => q !== query);
     this.saveQueryHistory();
     this.updateHistoryDisplay();
   }
   
   private loadQueryHistory(): void {
     try {
       const saved = localStorage.getItem('cortexkg-query-history');
       if (saved) {
         this.queryHistory = JSON.parse(saved);
       }
     } catch (error) {
       console.warn('Failed to load query history:', error);
       this.queryHistory = [];
     }
   }
   
   private saveQueryHistory(): void {
     try {
       localStorage.setItem('cortexkg-query-history', JSON.stringify(this.queryHistory));
     } catch (error) {
       console.warn('Failed to save query history:', error);
     }
   }
   
   public clearHistory(): void {
     this.queryHistory = [];
     this.saveQueryHistory();
     this.updateHistoryDisplay();
   }
   ```

## Expected Outputs
- Responsive query input with real-time autocomplete
- Advanced filtering system with concept types, activation thresholds
- Query history management with localStorage persistence
- Keyboard navigation support for suggestions
- Customizable result sorting and limiting
- Clean, accessible UI with proper ARIA labels

## Validation
1. Autocomplete suggestions appear within 200ms of typing
2. Filters correctly modify query results
3. Query history persists across browser sessions
4. Keyboard navigation works smoothly through suggestions
5. Interface remains responsive with complex queries

## Next Steps
- Create allocation interface component (micro-phase 9.25)
- Integrate query interface with canvas visualization