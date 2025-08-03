# Micro-Phase 9.25: Allocation Interface Component

## Objective
Build a comprehensive allocation interface with batch operations, validation, and real-time feedback for managing cortical column allocations.

## Prerequisites
- Completed micro-phase 9.24 (Query Interface)
- WASM module with allocation methods available
- Understanding of form validation and batch operations

## Task Description
Implement an intuitive allocation interface that supports single and batch allocations, provides validation feedback, and includes allocation history tracking.

## Specific Actions

1. **Create AllocationInterface class structure**:
   ```typescript
   // src/ui/AllocationInterface.ts
   import { CortexKGWasm } from 'cortexkg-wasm';
   
   export interface AllocationConfig {
     container: HTMLElement;
     visualizer?: any; // Reference to CorticalVisualizer for updates
     maxBatchSize?: number;
     enableHistory?: boolean;
     onAllocation?: (result: AllocationResult) => void;
     onError?: (error: AllocationError) => void;
   }
   
   export interface AllocationRequest {
     concepts: string[];
     relationships?: RelationshipRequest[];
     options: AllocationOptions;
   }
   
   export interface RelationshipRequest {
     source: string;
     target: string;
     type: string;
     weight?: number;
   }
   
   export interface AllocationOptions {
     threshold: number;
     maxColumns: number;
     preserveExisting: boolean;
     spreadActivation: boolean;
     priority: 'normal' | 'high' | 'batch';
   }
   
   export interface AllocationResult {
     success: boolean;
     allocatedColumns: number[];
     concepts: string[];
     metrics: AllocationMetrics;
     timestamp: Date;
   }
   
   export interface AllocationMetrics {
     allocatedCount: number;
     totalRequested: number;
     averageActivation: number;
     processingTime: number;
   }
   
   export interface AllocationError {
     type: 'validation' | 'capacity' | 'network' | 'unknown';
     message: string;
     details?: any;
   }
   
   export class AllocationInterface {
     private container: HTMLElement;
     private wasmModule: CortexKGWasm;
     private config: Required<AllocationConfig>;
     private formContainer: HTMLElement;
     private historyContainer: HTMLElement;
     private statusContainer: HTMLElement;
     private allocationHistory: AllocationResult[] = [];
     private currentRequest: AllocationRequest | null = null;
     private isProcessing = false;
     
     constructor(wasmModule: CortexKGWasm, config: AllocationConfig) {
       this.wasmModule = wasmModule;
       this.container = config.container;
       this.config = {
         container: config.container,
         visualizer: config.visualizer,
         maxBatchSize: config.maxBatchSize ?? 100,
         enableHistory: config.enableHistory ?? true,
         onAllocation: config.onAllocation ?? (() => {}),
         onError: config.onError ?? (() => {})
       };
       
       this.createInterface();
       this.setupEventHandlers();
       this.loadAllocationHistory();
     }
   }
   ```

2. **Implement interface creation and form structure**:
   ```typescript
   private createInterface(): void {
     this.container.innerHTML = `
       <div class="allocation-interface">
         <div class="allocation-form-section">
           <h3>Concept Allocation</h3>
           
           <div class="input-mode-toggle">
             <label class="mode-option">
               <input type="radio" name="input-mode" value="single" checked>
               Single Concept
             </label>
             <label class="mode-option">
               <input type="radio" name="input-mode" value="batch">
               Batch Allocation
             </label>
             <label class="mode-option">
               <input type="radio" name="input-mode" value="relationship">
               With Relationships
             </label>
           </div>
           
           <div class="form-container">
             <!-- Single concept form -->
             <div class="single-form form-mode active">
               <div class="form-group">
                 <label for="single-concept">Concept:</label>
                 <input type="text" id="single-concept" class="concept-input" 
                        placeholder="Enter concept name...">
                 <div class="concept-suggestions"></div>
               </div>
             </div>
             
             <!-- Batch allocation form -->
             <div class="batch-form form-mode">
               <div class="form-group">
                 <label for="batch-concepts">Concepts (one per line):</label>
                 <textarea id="batch-concepts" class="batch-textarea" 
                          placeholder="concept1&#10;concept2&#10;concept3..." 
                          rows="8"></textarea>
                 <div class="batch-info">
                   <span class="concept-count">0 concepts</span>
                   <span class="max-limit">Max: ${this.config.maxBatchSize}</span>
                 </div>
               </div>
               
               <div class="form-group">
                 <button type="button" class="validate-batch">Validate Batch</button>
                 <div class="validation-results"></div>
               </div>
             </div>
             
             <!-- Relationship form -->
             <div class="relationship-form form-mode">
               <div class="form-group">
                 <label for="source-concept">Source Concept:</label>
                 <input type="text" id="source-concept" class="concept-input">
               </div>
               
               <div class="form-group">
                 <label for="target-concept">Target Concept:</label>
                 <input type="text" id="target-concept" class="concept-input">
               </div>
               
               <div class="form-group">
                 <label for="relationship-type">Relationship Type:</label>
                 <select id="relationship-type">
                   <option value="relates_to">Relates To</option>
                   <option value="is_a">Is A</option>
                   <option value="part_of">Part Of</option>
                   <option value="similar_to">Similar To</option>
                   <option value="opposite_of">Opposite Of</option>
                   <option value="custom">Custom...</option>
                 </select>
                 <input type="text" id="custom-relationship" class="custom-input" 
                        placeholder="Custom relationship type" style="display: none;">
               </div>
               
               <div class="form-group">
                 <label for="relationship-weight">Weight (0-1):</label>
                 <input type="range" id="relationship-weight" min="0" max="1" 
                        step="0.1" value="0.8">
                 <span class="weight-value">0.8</span>
               </div>
             </div>
             
             <!-- Allocation options -->
             <div class="allocation-options">
               <h4>Allocation Options</h4>
               
               <div class="options-grid">
                 <div class="option-group">
                   <label for="threshold">Threshold:</label>
                   <input type="range" id="threshold" min="0" max="1" 
                          step="0.05" value="0.7">
                   <span class="threshold-value">0.7</span>
                 </div>
                 
                 <div class="option-group">
                   <label for="max-columns">Max Columns:</label>
                   <input type="number" id="max-columns" min="1" max="1024" value="10">
                 </div>
                 
                 <div class="option-group">
                   <label>
                     <input type="checkbox" id="preserve-existing" checked>
                     Preserve Existing
                   </label>
                 </div>
                 
                 <div class="option-group">
                   <label>
                     <input type="checkbox" id="spread-activation" checked>
                     Spread Activation
                   </label>
                 </div>
                 
                 <div class="option-group">
                   <label for="priority">Priority:</label>
                   <select id="priority">
                     <option value="normal" selected>Normal</option>
                     <option value="high">High</option>
                     <option value="batch">Batch</option>
                   </select>
                 </div>
               </div>
             </div>
             
             <div class="form-actions">
               <button type="button" class="allocate-button primary">Allocate</button>
               <button type="button" class="preview-button">Preview</button>
               <button type="button" class="clear-button">Clear</button>
             </div>
           </div>
         </div>
         
         <div class="status-section">
           <div class="status-container">
             <div class="status-header">Status</div>
             <div class="status-content">Ready for allocation</div>
             <div class="progress-bar">
               <div class="progress-fill"></div>
             </div>
           </div>
         </div>
         
         ${this.config.enableHistory ? `
         <div class="history-section">
           <div class="history-header">
             <h3>Allocation History</h3>
             <button class="clear-history">Clear History</button>
           </div>
           <div class="history-container"></div>
         </div>
         ` : ''}
       </div>
     `;
     
     // Get references to key elements
     this.formContainer = this.container.querySelector('.form-container') as HTMLElement;
     this.statusContainer = this.container.querySelector('.status-container') as HTMLElement;
     this.historyContainer = this.container.querySelector('.history-container') as HTMLElement;
   }
   ```

3. **Implement form mode switching and validation**:
   ```typescript
   private setupEventHandlers(): void {
     // Input mode switching
     this.container.querySelectorAll('input[name="input-mode"]').forEach(radio => {
       radio.addEventListener('change', (e) => {
         const mode = (e.target as HTMLInputElement).value;
         this.switchInputMode(mode);
       });
     });
     
     // Single concept autocomplete
     const singleInput = this.container.querySelector('#single-concept') as HTMLInputElement;
     if (singleInput) {
       this.setupAutocomplete(singleInput);
     }
     
     // Batch validation
     const validateButton = this.container.querySelector('.validate-batch') as HTMLButtonElement;
     validateButton?.addEventListener('click', () => {
       this.validateBatchConcepts();
     });
     
     // Batch concept counting
     const batchTextarea = this.container.querySelector('#batch-concepts') as HTMLTextAreaElement;
     batchTextarea?.addEventListener('input', () => {
       this.updateBatchCount();
     });
     
     // Relationship type handling
     const relationshipSelect = this.container.querySelector('#relationship-type') as HTMLSelectElement;
     const customInput = this.container.querySelector('#custom-relationship') as HTMLInputElement;
     
     relationshipSelect?.addEventListener('change', () => {
       customInput.style.display = relationshipSelect.value === 'custom' ? 'block' : 'none';
     });
     
     // Range input updates
     this.setupRangeInputs();
     
     // Action buttons
     this.container.querySelector('.allocate-button')?.addEventListener('click', () => {
       this.executeAllocation();
     });
     
     this.container.querySelector('.preview-button')?.addEventListener('click', () => {
       this.previewAllocation();
     });
     
     this.container.querySelector('.clear-button')?.addEventListener('click', () => {
       this.clearForm();
     });
     
     // History actions
     if (this.config.enableHistory) {
       this.container.querySelector('.clear-history')?.addEventListener('click', () => {
         this.clearHistory();
       });
     }
   }
   
   private switchInputMode(mode: string): void {
     // Hide all form modes
     this.container.querySelectorAll('.form-mode').forEach(form => {
       form.classList.remove('active');
     });
     
     // Show selected mode
     const targetForm = this.container.querySelector(`.${mode}-form`);
     targetForm?.classList.add('active');
     
     // Update validation and options accordingly
     this.updateFormValidation();
   }
   
   private setupRangeInputs(): void {
     // Threshold slider
     const thresholdSlider = this.container.querySelector('#threshold') as HTMLInputElement;
     const thresholdValue = this.container.querySelector('.threshold-value') as HTMLElement;
     
     thresholdSlider?.addEventListener('input', () => {
       thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
     });
     
     // Relationship weight slider
     const weightSlider = this.container.querySelector('#relationship-weight') as HTMLInputElement;
     const weightValue = this.container.querySelector('.weight-value') as HTMLElement;
     
     weightSlider?.addEventListener('input', () => {
       weightValue.textContent = parseFloat(weightSlider.value).toFixed(1);
     });
   }
   
   private updateBatchCount(): void {
     const textarea = this.container.querySelector('#batch-concepts') as HTMLTextAreaElement;
     const countDisplay = this.container.querySelector('.concept-count') as HTMLElement;
     
     const concepts = textarea.value.split('\n').filter(line => line.trim().length > 0);
     countDisplay.textContent = `${concepts.length} concepts`;
     
     // Highlight if over limit
     if (concepts.length > this.config.maxBatchSize) {
       countDisplay.classList.add('over-limit');
     } else {
       countDisplay.classList.remove('over-limit');
     }
   }
   ```

4. **Implement allocation execution and validation**:
   ```typescript
   private async executeAllocation(): Promise<void> {
     if (this.isProcessing) return;
     
     try {
       this.isProcessing = true;
       this.updateStatus('Processing allocation...', 'processing');
       
       const request = this.buildAllocationRequest();
       if (!request) {
         throw new Error('Invalid allocation request');
       }
       
       const startTime = performance.now();
       
       // Execute allocation based on mode
       let result: AllocationResult;
       
       if (request.concepts.length === 1 && !request.relationships?.length) {
         result = await this.executeSingleAllocation(request);
       } else if (request.relationships?.length) {
         result = await this.executeRelationshipAllocation(request);
       } else {
         result = await this.executeBatchAllocation(request);
       }
       
       const processingTime = performance.now() - startTime;
       result.metrics.processingTime = processingTime;
       
       // Update visualization if available
       if (this.config.visualizer && result.success) {
         this.updateVisualization(result);
       }
       
       // Add to history
       this.addToHistory(result);
       
       // Update UI
       this.updateStatus(`Allocated ${result.metrics.allocatedCount} columns successfully`, 'success');
       this.config.onAllocation(result);
       
     } catch (error) {
       const allocationError: AllocationError = {
         type: 'unknown',
         message: error instanceof Error ? error.message : 'Unknown error occurred',
         details: error
       };
       
       this.updateStatus(`Allocation failed: ${allocationError.message}`, 'error');
       this.config.onError(allocationError);
       
     } finally {
       this.isProcessing = false;
     }
   }
   
   private buildAllocationRequest(): AllocationRequest | null {
     const activeMode = this.container.querySelector('.form-mode.active') as HTMLElement;
     const mode = activeMode.classList.contains('single-form') ? 'single' :
                  activeMode.classList.contains('batch-form') ? 'batch' : 'relationship';
     
     const options = this.buildAllocationOptions();
     let concepts: string[] = [];
     let relationships: RelationshipRequest[] = [];
     
     try {
       if (mode === 'single') {
         const input = this.container.querySelector('#single-concept') as HTMLInputElement;
         const concept = input.value.trim();
         if (!concept) throw new Error('Concept name is required');
         concepts = [concept];
         
       } else if (mode === 'batch') {
         const textarea = this.container.querySelector('#batch-concepts') as HTMLTextAreaElement;
         concepts = textarea.value.split('\n')
           .map(line => line.trim())
           .filter(line => line.length > 0);
         
         if (concepts.length === 0) throw new Error('At least one concept is required');
         if (concepts.length > this.config.maxBatchSize) {
           throw new Error(`Batch size exceeds maximum of ${this.config.maxBatchSize}`);
         }
         
       } else if (mode === 'relationship') {
         const sourceInput = this.container.querySelector('#source-concept') as HTMLInputElement;
         const targetInput = this.container.querySelector('#target-concept') as HTMLInputElement;
         const typeSelect = this.container.querySelector('#relationship-type') as HTMLSelectElement;
         const customInput = this.container.querySelector('#custom-relationship') as HTMLInputElement;
         const weightSlider = this.container.querySelector('#relationship-weight') as HTMLInputElement;
         
         const source = sourceInput.value.trim();
         const target = targetInput.value.trim();
         let relType = typeSelect.value;
         
         if (!source || !target) throw new Error('Both source and target concepts are required');
         if (relType === 'custom') {
           relType = customInput.value.trim();
           if (!relType) throw new Error('Custom relationship type is required');
         }
         
         concepts = [source, target];
         relationships = [{
           source,
           target,
           type: relType,
           weight: parseFloat(weightSlider.value)
         }];
       }
       
       return { concepts, relationships, options };
       
     } catch (error) {
       this.updateStatus(`Validation error: ${error.message}`, 'error');
       return null;
     }
   }
   
   private buildAllocationOptions(): AllocationOptions {
     const threshold = parseFloat((this.container.querySelector('#threshold') as HTMLInputElement).value);
     const maxColumns = parseInt((this.container.querySelector('#max-columns') as HTMLInputElement).value);
     const preserveExisting = (this.container.querySelector('#preserve-existing') as HTMLInputElement).checked;
     const spreadActivation = (this.container.querySelector('#spread-activation') as HTMLInputElement).checked;
     const priority = (this.container.querySelector('#priority') as HTMLSelectElement).value as AllocationOptions['priority'];
     
     return {
       threshold,
       maxColumns,
       preserveExisting,
       spreadActivation,
       priority
     };
   }
   
   private async executeSingleAllocation(request: AllocationRequest): Promise<AllocationResult> {
     const concept = request.concepts[0];
     const columnId = this.wasmModule.allocate_concept(concept, request.options.threshold);
     
     if (columnId < 0) {
       throw new Error('Failed to allocate concept - no available columns');
     }
     
     return {
       success: true,
       allocatedColumns: [columnId],
       concepts: [concept],
       metrics: {
         allocatedCount: 1,
         totalRequested: 1,
         averageActivation: request.options.threshold,
         processingTime: 0 // Will be set by caller
       },
       timestamp: new Date()
     };
   }
   
   private async executeBatchAllocation(request: AllocationRequest): Promise<AllocationResult> {
     const results = this.wasmModule.allocate_concepts_batch(
       request.concepts,
       request.options.threshold,
       request.options.maxColumns
     );
     
     const allocatedColumns = results.allocated_columns || [];
     const totalActivation = results.total_activation || 0;
     
     return {
       success: allocatedColumns.length > 0,
       allocatedColumns,
       concepts: request.concepts,
       metrics: {
         allocatedCount: allocatedColumns.length,
         totalRequested: request.concepts.length,
         averageActivation: allocatedColumns.length > 0 ? totalActivation / allocatedColumns.length : 0,
         processingTime: 0
       },
       timestamp: new Date()
     };
   }
   
   private async executeRelationshipAllocation(request: AllocationRequest): Promise<AllocationResult> {
     const results: number[] = [];
     
     // Allocate concepts first
     for (const concept of request.concepts) {
       const columnId = this.wasmModule.allocate_concept(concept, request.options.threshold);
       if (columnId >= 0) {
         results.push(columnId);
       }
     }
     
     // Create relationships
     if (request.relationships && results.length >= 2) {
       for (const rel of request.relationships) {
         this.wasmModule.create_relationship(rel.source, rel.target, rel.type, rel.weight || 0.8);
       }
     }
     
     return {
       success: results.length > 0,
       allocatedColumns: results,
       concepts: request.concepts,
       metrics: {
         allocatedCount: results.length,
         totalRequested: request.concepts.length,
         averageActivation: request.options.threshold,
         processingTime: 0
       },
       timestamp: new Date()
     };
   }
   ```

5. **Implement preview and batch validation**:
   ```typescript
   private async previewAllocation(): Promise<void> {
     const request = this.buildAllocationRequest();
     if (!request) return;
     
     try {
       this.updateStatus('Generating preview...', 'processing');
       
       // Check available capacity
       const availableColumns = this.wasmModule.get_available_column_count();
       const requiredColumns = Math.min(request.concepts.length, request.options.maxColumns);
       
       if (requiredColumns > availableColumns) {
         this.updateStatus(`Warning: Only ${availableColumns} columns available for ${requiredColumns} concepts`, 'warning');
       }
       
       // Generate allocation preview
       const preview = this.generateAllocationPreview(request);
       this.displayPreview(preview);
       
       this.updateStatus('Preview generated successfully', 'success');
       
     } catch (error) {
       this.updateStatus(`Preview failed: ${error.message}`, 'error');
     }
   }
   
   private async validateBatchConcepts(): Promise<void> {
     const textarea = this.container.querySelector('#batch-concepts') as HTMLTextAreaElement;
     const resultsContainer = this.container.querySelector('.validation-results') as HTMLElement;
     
     const concepts = textarea.value.split('\n')
       .map(line => line.trim())
       .filter(line => line.length > 0);
     
     if (concepts.length === 0) {
       resultsContainer.innerHTML = '<div class="validation-empty">No concepts to validate</div>';
       return;
     }
     
     this.updateStatus('Validating batch concepts...', 'processing');
     
     const validation = {
       total: concepts.length,
       valid: 0,
       duplicates: 0,
       existing: 0,
       invalid: []
     };
     
     const seen = new Set();
     const existingConcepts = new Set(this.wasmModule.get_all_concepts());
     
     for (const concept of concepts) {
       if (seen.has(concept)) {
         validation.duplicates++;
         continue;
       }
       seen.add(concept);
       
       if (existingConcepts.has(concept)) {
         validation.existing++;
       }
       
       if (this.validateConceptName(concept)) {
         validation.valid++;
       } else {
         validation.invalid.push(concept);
       }
     }
     
     // Display validation results
     const html = `
       <div class="validation-summary">
         <div class="validation-stat valid">✓ ${validation.valid} valid</div>
         <div class="validation-stat existing">⚠ ${validation.existing} existing</div>
         <div class="validation-stat duplicates">⚠ ${validation.duplicates} duplicates</div>
         <div class="validation-stat invalid">✗ ${validation.invalid.length} invalid</div>
       </div>
       ${validation.invalid.length > 0 ? `
         <div class="invalid-concepts">
           <strong>Invalid concepts:</strong>
           <ul>${validation.invalid.map(c => `<li>${c}</li>`).join('')}</ul>
         </div>
       ` : ''}
     `;
     
     resultsContainer.innerHTML = html;
     this.updateStatus('Batch validation complete', 'success');
   }
   
   private validateConceptName(name: string): boolean {
     // Check for valid concept name format
     if (name.length < 2 || name.length > 100) return false;
     if (!/^[a-zA-Z0-9\s\-_]+$/.test(name)) return false;
     return true;
   }
   
   private generateAllocationPreview(request: AllocationRequest): any {
     return {
       conceptCount: request.concepts.length,
       estimatedColumns: Math.min(request.concepts.length, request.options.maxColumns),
       threshold: request.options.threshold,
       options: request.options,
       relationships: request.relationships?.length || 0
     };
   }
   
   private displayPreview(preview: any): void {
     // Implementation would show preview in a modal or dedicated section
     console.log('Allocation Preview:', preview);
   }
   ```

6. **Implement history management and UI updates**:
   ```typescript
   private updateStatus(message: string, type: 'ready' | 'processing' | 'success' | 'error' | 'warning'): void {
     const statusContent = this.statusContainer.querySelector('.status-content') as HTMLElement;
     const progressBar = this.statusContainer.querySelector('.progress-fill') as HTMLElement;
     
     statusContent.textContent = message;
     statusContent.className = `status-content ${type}`;
     
     // Update progress bar
     if (type === 'processing') {
       progressBar.style.width = '50%';
       progressBar.classList.add('animated');
     } else {
       progressBar.style.width = type === 'success' ? '100%' : '0%';
       progressBar.classList.remove('animated');
     }
   }
   
   private addToHistory(result: AllocationResult): void {
     if (!this.config.enableHistory) return;
     
     this.allocationHistory.unshift(result);
     
     // Limit history size
     if (this.allocationHistory.length > 50) {
       this.allocationHistory = this.allocationHistory.slice(0, 50);
     }
     
     this.saveAllocationHistory();
     this.updateHistoryDisplay();
   }
   
   private updateHistoryDisplay(): void {
     if (!this.historyContainer) return;
     
     const html = this.allocationHistory.map((result, index) => `
       <div class="history-item ${result.success ? 'success' : 'failed'}">
         <div class="history-header">
           <span class="history-time">${result.timestamp.toLocaleTimeString()}</span>
           <span class="history-status ${result.success ? 'success' : 'failed'}">
             ${result.success ? '✓' : '✗'}
           </span>
         </div>
         <div class="history-content">
           <div class="history-concepts">${result.concepts.slice(0, 3).join(', ')}${result.concepts.length > 3 ? '...' : ''}</div>
           <div class="history-metrics">
             ${result.metrics.allocatedCount}/${result.metrics.totalRequested} allocated
             (${(result.metrics.processingTime).toFixed(0)}ms)
           </div>
         </div>
         <div class="history-actions">
           <button class="repeat-allocation" data-index="${index}">Repeat</button>
           <button class="view-details" data-index="${index}">Details</button>
         </div>
       </div>
     `).join('');
     
     this.historyContainer.innerHTML = html || '<div class="no-history">No allocation history</div>';
     
     // Add event handlers
     this.historyContainer.querySelectorAll('.repeat-allocation').forEach(button => {
       button.addEventListener('click', (e) => {
         const index = parseInt((e.target as HTMLElement).getAttribute('data-index')!);
         this.repeatAllocation(this.allocationHistory[index]);
       });
     });
   }
   
   private repeatAllocation(result: AllocationResult): void {
     // Fill form with previous allocation data
     if (result.concepts.length === 1) {
       const radio = this.container.querySelector('input[value="single"]') as HTMLInputElement;
       radio.checked = true;
       this.switchInputMode('single');
       
       const input = this.container.querySelector('#single-concept') as HTMLInputElement;
       input.value = result.concepts[0];
     } else {
       const radio = this.container.querySelector('input[value="batch"]') as HTMLInputElement;
       radio.checked = true;
       this.switchInputMode('batch');
       
       const textarea = this.container.querySelector('#batch-concepts') as HTMLTextAreaElement;
       textarea.value = result.concepts.join('\n');
     }
   }
   
   private clearForm(): void {
     this.container.querySelectorAll('input[type="text"], textarea').forEach(input => {
       (input as HTMLInputElement | HTMLTextAreaElement).value = '';
     });
     
     // Reset to defaults
     (this.container.querySelector('#threshold') as HTMLInputElement).value = '0.7';
     (this.container.querySelector('#max-columns') as HTMLInputElement).value = '10';
     (this.container.querySelector('#preserve-existing') as HTMLInputElement).checked = true;
     (this.container.querySelector('#spread-activation') as HTMLInputElement).checked = true;
     
     this.updateStatus('Form cleared', 'ready');
   }
   
   private clearHistory(): void {
     this.allocationHistory = [];
     this.saveAllocationHistory();
     this.updateHistoryDisplay();
   }
   
   private loadAllocationHistory(): void {
     try {
       const saved = localStorage.getItem('cortexkg-allocation-history');
       if (saved) {
         const data = JSON.parse(saved);
         this.allocationHistory = data.map((item: any) => ({
           ...item,
           timestamp: new Date(item.timestamp)
         }));
         this.updateHistoryDisplay();
       }
     } catch (error) {
       console.warn('Failed to load allocation history:', error);
     }
   }
   
   private saveAllocationHistory(): void {
     try {
       localStorage.setItem('cortexkg-allocation-history', JSON.stringify(this.allocationHistory));
     } catch (error) {
       console.warn('Failed to save allocation history:', error);
     }
   }
   
   public getMetrics(): AllocationMetrics | null {
     return this.allocationHistory.length > 0 ? this.allocationHistory[0].metrics : null;
   }
   
   public getCurrentRequest(): AllocationRequest | null {
     return this.currentRequest;
   }
   ```

## Expected Outputs
- Multi-mode allocation interface (single, batch, relationship)
- Real-time validation and feedback system
- Batch concept validation with duplicate detection
- Allocation history with repeat functionality
- Progress tracking and error handling
- Integration with cortical visualizer updates

## Validation
1. Single concept allocation completes within 100ms
2. Batch validation identifies duplicates and invalid names
3. Form validation prevents invalid submissions
4. History persists correctly across sessions
5. Interface remains responsive during large batch operations

## Next Steps
- Create canvas setup component (micro-phase 9.26)
- Integrate allocation interface with query interface