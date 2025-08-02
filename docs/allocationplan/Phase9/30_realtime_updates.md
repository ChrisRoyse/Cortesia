# Micro-Phase 9.30: Real-time Updates Component

## Objective
Create an efficient real-time update system with delta rendering, change tracking, and optimized data synchronization for neuromorphic visualizations.

## Prerequisites
- Completed micro-phase 9.29 (Touch Interactions)
- TouchInteractionManager class available
- Understanding of change detection and delta rendering techniques

## Task Description
Implement a sophisticated real-time update system that efficiently tracks changes, applies delta updates, and maintains synchronized state between WASM and UI components.

## Specific Actions

1. **Create RealtimeUpdateManager class structure**:
   ```typescript
   // src/ui/RealtimeUpdateManager.ts
   import { CortexKGWasm } from 'cortexkg-wasm';
   import { CanvasManager } from './CanvasManager';
   import { ColumnRenderer, ColumnState } from './ColumnRenderer';
   import { ActivationAnimator } from './ActivationAnimator';
   import { TouchInteractionManager } from './TouchInteractionManager';
   
   export interface UpdateConfig {
     wasmModule: CortexKGWasm;
     canvasManager: CanvasManager;
     columnRenderer: ColumnRenderer;
     activationAnimator: ActivationAnimator;
     touchManager: TouchInteractionManager;
     updateInterval?: number;
     batchSize?: number;
     enableDeltaRendering?: boolean;
     enableChangeTracking?: boolean;
     maxUpdateQueue?: number;
     debugMode?: boolean;
   }
   
   export interface ColumnUpdate {
     id: number;
     activation?: number;
     allocated?: boolean;
     position?: [number, number];
     size?: number;
     connections?: number[];
     metadata?: any;
     timestamp: number;
   }
   
   export interface SystemUpdate {
     type: 'column' | 'relationship' | 'allocation' | 'query' | 'system';
     data: any;
     timestamp: number;
     source: 'wasm' | 'ui' | 'external';
   }
   
   export interface ChangeSet {
     additions: ColumnUpdate[];
     modifications: ColumnUpdate[];
     deletions: number[];
     relationshipChanges: RelationshipUpdate[];
     timestamp: number;
   }
   
   export interface RelationshipUpdate {
     source: number;
     target: number;
     type: string;
     weight: number;
     operation: 'add' | 'remove' | 'modify';
     timestamp: number;
   }
   
   export interface RenderFrame {
     frameId: number;
     timestamp: number;
     changes: ChangeSet;
     dirty: boolean;
     rendered: boolean;
   }
   
   export interface UpdateStats {
     updatesPerSecond: number;
     averageUpdateTime: number;
     queueSize: number;
     droppedUpdates: number;
     lastUpdateTime: number;
     totalUpdates: number;
   }
   
   export class RealtimeUpdateManager {
     private wasmModule: CortexKGWasm;
     private canvasManager: CanvasManager;
     private columnRenderer: ColumnRenderer;
     private activationAnimator: ActivationAnimator;
     private touchManager: TouchInteractionManager;
     private config: Required<UpdateConfig>;
     
     private isRunning = false;
     private updateInterval: number | null = null;
     private renderFrame: number | null = null;
     
     private columnStates: Map<number, ColumnState> = new Map();
     private previousStates: Map<number, ColumnState> = new Map();
     private pendingUpdates: SystemUpdate[] = [];
     private updateQueue: ChangeSet[] = [];
     private currentFrame: RenderFrame | null = null;
     private frameCounter = 0;
     
     private stats: UpdateStats = {
       updatesPerSecond: 0,
       averageUpdateTime: 0,
       queueSize: 0,
       droppedUpdates: 0,
       lastUpdateTime: 0,
       totalUpdates: 0
     };
     
     private lastStatsUpdate = 0;
     private updateTimes: number[] = [];
     private isDirty = false;
     
     // Event callbacks
     public onUpdate?: (changeSet: ChangeSet) => void;
     public onError?: (error: Error) => void;
     public onStatsUpdate?: (stats: UpdateStats) => void;
     
     constructor(config: UpdateConfig) {
       this.wasmModule = config.wasmModule;
       this.canvasManager = config.canvasManager;
       this.columnRenderer = config.columnRenderer;
       this.activationAnimator = config.activationAnimator;
       this.touchManager = config.touchManager;
       
       this.config = {
         wasmModule: config.wasmModule,
         canvasManager: config.canvasManager,
         columnRenderer: config.columnRenderer,
         activationAnimator: config.activationAnimator,
         touchManager: config.touchManager,
         updateInterval: config.updateInterval ?? 16, // 60fps
         batchSize: config.batchSize ?? 50,
         enableDeltaRendering: config.enableDeltaRendering ?? true,
         enableChangeTracking: config.enableChangeTracking ?? true,
         maxUpdateQueue: config.maxUpdateQueue ?? 100,
         debugMode: config.debugMode ?? false
       };
       
       this.initializeStates();
       this.setupEventHandlers();
     }
   }
   ```

2. **Implement initialization and state management**:
   ```typescript
   private initializeStates(): void {
     try {
       // Get initial column states from WASM
       const columnCount = this.wasmModule.column_count;
       
       for (let i = 0; i < columnCount; i++) {
         const columnData = this.wasmModule.get_column_state(i);
         
         if (columnData) {
           const state: ColumnState = {
             id: i,
             position: [columnData.position_x || 0, columnData.position_y || 0],
             size: columnData.size || 20,
             activation: columnData.activation || 0,
             allocated: columnData.allocated || false,
             selected: false,
             hovered: false,
             connections: columnData.connections || [],
             metadata: columnData.metadata
           };
           
           this.columnStates.set(i, state);
           this.previousStates.set(i, { ...state });
         }
       }
       
       // Initialize column renderer with current states
       const initialStates = Array.from(this.columnStates.values());
       this.columnRenderer.setColumns(initialStates);
       
       if (this.config.debugMode) {
         console.log(`Initialized ${this.columnStates.size} column states`);
       }
       
     } catch (error) {
       console.error('Failed to initialize states:', error);
       if (this.onError) {
         this.onError(error as Error);
       }
     }
   }
   
   private setupEventHandlers(): void {
     // Listen for WASM events if supported
     if ('addEventListener' in this.wasmModule) {
       (this.wasmModule as any).addEventListener('column_changed', (event: any) => {
         this.handleWasmUpdate(event.detail);
       });
       
       (this.wasmModule as any).addEventListener('allocation_changed', (event: any) => {
         this.handleAllocationUpdate(event.detail);
       });
       
       (this.wasmModule as any).addEventListener('activation_changed', (event: any) => {
         this.handleActivationUpdate(event.detail);
       });
     }
     
     // Listen for touch manager selection changes
     this.touchManager.onSelectionChanged = (selectedColumns: number[]) => {
       this.handleSelectionUpdate(selectedColumns);
     };
     
     // Listen for canvas resize
     this.canvasManager.canvasElement.addEventListener('canvasResize', () => {
       this.handleCanvasResize();
     });
   }
   
   public start(): void {
     if (this.isRunning) return;
     
     this.isRunning = true;
     this.scheduleUpdate();
     this.scheduleRender();
     
     if (this.config.debugMode) {
       console.log('RealtimeUpdateManager started');
     }
   }
   
   public stop(): void {
     if (!this.isRunning) return;
     
     this.isRunning = false;
     
     if (this.updateInterval) {
       clearInterval(this.updateInterval);
       this.updateInterval = null;
     }
     
     if (this.renderFrame) {
       cancelAnimationFrame(this.renderFrame);
       this.renderFrame = null;
     }
     
     if (this.config.debugMode) {
       console.log('RealtimeUpdateManager stopped');
     }
   }
   
   private scheduleUpdate(): void {
     this.updateInterval = window.setInterval(() => {
       if (this.isRunning) {
         this.performUpdate();
       }
     }, this.config.updateInterval);
   }
   
   private scheduleRender(): void {
     const render = () => {
       if (this.isRunning) {
         this.performRender();
         this.renderFrame = requestAnimationFrame(render);
       }
     };
     
     this.renderFrame = requestAnimationFrame(render);
   }
   ```

3. **Implement change detection and delta updates**:
   ```typescript
   private performUpdate(): void {
     const startTime = performance.now();
     
     try {
       // Poll WASM for changes
       const changes = this.detectChanges();
       
       if (changes.additions.length > 0 || changes.modifications.length > 0 || 
           changes.deletions.length > 0 || changes.relationshipChanges.length > 0) {
         
         // Process changes
         this.processChangeSet(changes);
         
         // Add to update queue
         if (this.updateQueue.length >= this.config.maxUpdateQueue) {
           this.updateQueue.shift(); // Remove oldest
           this.stats.droppedUpdates++;
         }
         
         this.updateQueue.push(changes);
         this.isDirty = true;
         
         // Trigger callback
         if (this.onUpdate) {
           this.onUpdate(changes);
         }
       }
       
       // Process pending UI updates
       this.processPendingUpdates();
       
       // Update statistics
       const updateTime = performance.now() - startTime;
       this.updateStats(updateTime);
       
     } catch (error) {
       console.error('Update error:', error);
       if (this.onError) {
         this.onError(error as Error);
       }
     }
   }
   
   private detectChanges(): ChangeSet {
     const timestamp = performance.now();
     const changes: ChangeSet = {
       additions: [],
       modifications: [],
       deletions: [],
       relationshipChanges: [],
       timestamp
     };
     
     if (!this.config.enableChangeTracking) {
       return changes;
     }
     
     try {
       // Check for column changes
       const currentColumnCount = this.wasmModule.column_count;
       
       // Detect new columns
       for (let i = this.columnStates.size; i < currentColumnCount; i++) {
         const columnData = this.wasmModule.get_column_state(i);
         if (columnData) {
           const newState: ColumnState = {
             id: i,
             position: [columnData.position_x || 0, columnData.position_y || 0],
             size: columnData.size || 20,
             activation: columnData.activation || 0,
             allocated: columnData.allocated || false,
             selected: false,
             hovered: false,
             connections: columnData.connections || [],
             metadata: columnData.metadata
           };
           
           this.columnStates.set(i, newState);
           changes.additions.push({
             id: i,
             activation: newState.activation,
             allocated: newState.allocated,
             position: newState.position,
             size: newState.size,
             connections: newState.connections,
             timestamp
           });
         }
       }
       
       // Detect modifications
       for (const [id, currentState] of this.columnStates) {
         const previousState = this.previousStates.get(id);
         if (!previousState) continue;
         
         const columnData = this.wasmModule.get_column_state(id);
         if (!columnData) continue;
         
         const hasChanges = this.detectColumnChanges(currentState, columnData);
         
         if (hasChanges) {
           // Update current state
           currentState.activation = columnData.activation || 0;
           currentState.allocated = columnData.allocated || false;
           currentState.connections = columnData.connections || [];
           
           changes.modifications.push({
             id,
             activation: currentState.activation,
             allocated: currentState.allocated,
             connections: currentState.connections,
             timestamp
           });
         }
       }
       
       // Detect relationship changes
       const relationshipChanges = this.detectRelationshipChanges();
       changes.relationshipChanges = relationshipChanges;
       
       // Update previous states
       this.updatePreviousStates();
       
     } catch (error) {
       console.error('Change detection error:', error);
     }
     
     return changes;
   }
   
   private detectColumnChanges(currentState: ColumnState, wasmData: any): boolean {
     const previousState = this.previousStates.get(currentState.id);
     if (!previousState) return true;
     
     return (
       Math.abs(currentState.activation - (wasmData.activation || 0)) > 0.001 ||
       currentState.allocated !== (wasmData.allocated || false) ||
       !this.arraysEqual(currentState.connections, wasmData.connections || [])
     );
   }
   
   private detectRelationshipChanges(): RelationshipUpdate[] {
     const changes: RelationshipUpdate[] = [];
     const timestamp = performance.now();
     
     try {
       // Get current relationships from WASM
       const currentRelationships = this.wasmModule.get_all_relationships();
       
       // Compare with cached relationships
       // This would require implementing relationship caching
       // For now, return empty array
       
     } catch (error) {
       console.error('Relationship change detection error:', error);
     }
     
     return changes;
   }
   
   private processChangeSet(changes: ChangeSet): void {
     // Process additions
     for (const addition of changes.additions) {
       const state = this.columnStates.get(addition.id);
       if (state) {
         // Trigger addition animation
         this.activationAnimator.animateGlow(addition.id, 0.5, 1000);
       }
     }
     
     // Process modifications
     for (const modification of changes.modifications) {
       const state = this.columnStates.get(modification.id);
       if (state) {
         // Update state
         if (modification.activation !== undefined) {
           const oldActivation = state.activation;
           state.activation = modification.activation;
           
           // Trigger activation animation if significant change
           if (Math.abs(modification.activation - oldActivation) > 0.1) {
             this.activationAnimator.animateActivation(
               modification.id,
               modification.activation,
               300
             );
           }
         }
         
         if (modification.allocated !== undefined) {
           state.allocated = modification.allocated;
           
           // Trigger allocation animation
           if (modification.allocated) {
             this.activationAnimator.animatePulse(modification.id, 0.3, 1, 500);
           }
         }
         
         if (modification.connections !== undefined) {
           state.connections = modification.connections;
         }
       }
     }
     
     // Process deletions
     for (const deletionId of changes.deletions) {
       this.columnStates.delete(deletionId);
       this.previousStates.delete(deletionId);
     }
     
     // Process relationship changes
     for (const relationshipChange of changes.relationshipChanges) {
       this.processRelationshipChange(relationshipChange);
     }
   }
   ```

4. **Implement rendering optimization and delta rendering**:
   ```typescript
   private performRender(): void {
     if (!this.isDirty && this.config.enableDeltaRendering) {
       return; // Skip render if nothing changed
     }
     
     const startTime = performance.now();
     
     try {
       // Create render frame
       const frame: RenderFrame = {
         frameId: this.frameCounter++,
         timestamp: startTime,
         changes: this.getLatestChanges(),
         dirty: this.isDirty,
         rendered: false
       };
       
       this.currentFrame = frame;
       
       if (this.config.enableDeltaRendering) {
         this.performDeltaRender(frame);
       } else {
         this.performFullRender(frame);
       }
       
       frame.rendered = true;
       this.isDirty = false;
       
       // Clear processed updates
       this.updateQueue = [];
       
       if (this.config.debugMode) {
         const renderTime = performance.now() - startTime;
         console.log(`Frame ${frame.frameId} rendered in ${renderTime.toFixed(2)}ms`);
       }
       
     } catch (error) {
       console.error('Render error:', error);
       if (this.onError) {
         this.onError(error as Error);
       }
     }
   }
   
   private performDeltaRender(frame: RenderFrame): void {
     const changes = frame.changes;
     
     // Only render changed columns
     const changedColumns: ColumnState[] = [];
     
     // Collect changed columns
     for (const addition of changes.additions) {
       const state = this.columnStates.get(addition.id);
       if (state) changedColumns.push(state);
     }
     
     for (const modification of changes.modifications) {
       const state = this.columnStates.get(modification.id);
       if (state) changedColumns.push(state);
     }
     
     if (changedColumns.length > 0) {
       // Render only changed columns
       this.columnRenderer.renderColumns(changedColumns);
     }
     
     // Update column renderer state
     const allStates = Array.from(this.columnStates.values());
     this.columnRenderer.setColumns(allStates);
   }
   
   private performFullRender(frame: RenderFrame): void {
     // Render all columns
     const allStates = Array.from(this.columnStates.values());
     this.columnRenderer.renderColumns(allStates);
   }
   
   private getLatestChanges(): ChangeSet {
     if (this.updateQueue.length === 0) {
       return {
         additions: [],
         modifications: [],
         deletions: [],
         relationshipChanges: [],
         timestamp: performance.now()
       };
     }
     
     // Merge all changes in queue
     const merged: ChangeSet = {
       additions: [],
       modifications: [],
       deletions: [],
       relationshipChanges: [],
       timestamp: performance.now()
     };
     
     for (const changeSet of this.updateQueue) {
       merged.additions.push(...changeSet.additions);
       merged.modifications.push(...changeSet.modifications);
       merged.deletions.push(...changeSet.deletions);
       merged.relationshipChanges.push(...changeSet.relationshipChanges);
     }
     
     return merged;
   }
   ```

5. **Implement event handlers and utility methods**:
   ```typescript
   private handleWasmUpdate(data: any): void {
     const update: SystemUpdate = {
       type: 'column',
       data,
       timestamp: performance.now(),
       source: 'wasm'
     };
     
     this.pendingUpdates.push(update);
   }
   
   private handleAllocationUpdate(data: any): void {
     const update: SystemUpdate = {
       type: 'allocation',
       data,
       timestamp: performance.now(),
       source: 'wasm'
     };
     
     this.pendingUpdates.push(update);
     
     // Trigger animation for allocated column
     if (data.columnId !== undefined) {
       this.activationAnimator.animatePulse(data.columnId, 0.5, 1, 800);
     }
   }
   
   private handleActivationUpdate(data: any): void {
     const update: SystemUpdate = {
       type: 'column',
       data,
       timestamp: performance.now(),
       source: 'wasm'
     };
     
     this.pendingUpdates.push(update);
     
     // Trigger spreading activation if applicable
     if (data.spreading && data.targetColumns) {
       this.activationAnimator.animateSpreadingActivation(
         data.sourceColumn,
         data.targetColumns,
         data.speed || 200,
         data.intensity || 0.8
       );
     }
   }
   
   private handleSelectionUpdate(selectedColumns: number[]): void {
     // Update selection states
     for (const [id, state] of this.columnStates) {
       const wasSelected = state.selected;
       const isSelected = selectedColumns.includes(id);
       
       if (wasSelected !== isSelected) {
         state.selected = isSelected;
         this.isDirty = true;
         
         // Trigger selection animation
         if (isSelected) {
           this.activationAnimator.animateGlow(id, 0.3, 500);
         }
       }
     }
   }
   
   private handleCanvasResize(): void {
     // Recalculate column positions if needed
     this.isDirty = true;
   }
   
   private processPendingUpdates(): void {
     const batchSize = Math.min(this.pendingUpdates.length, this.config.batchSize);
     
     for (let i = 0; i < batchSize; i++) {
       const update = this.pendingUpdates.shift();
       if (update) {
         this.processSystemUpdate(update);
       }
     }
   }
   
   private processSystemUpdate(update: SystemUpdate): void {
     switch (update.type) {
       case 'column':
         this.processColumnUpdate(update.data);
         break;
       case 'allocation':
         this.processAllocationUpdate(update.data);
         break;
       case 'relationship':
         this.processRelationshipUpdate(update.data);
         break;
       default:
         console.warn('Unknown update type:', update.type);
     }
   }
   
   private processColumnUpdate(data: any): void {
     const state = this.columnStates.get(data.id);
     if (state) {
       if (data.activation !== undefined) {
         state.activation = data.activation;
       }
       if (data.allocated !== undefined) {
         state.allocated = data.allocated;
       }
       this.isDirty = true;
     }
   }
   
   private processAllocationUpdate(data: any): void {
     // Handle allocation-specific updates
     this.isDirty = true;
   }
   
   private processRelationshipUpdate(data: any): void {
     // Handle relationship updates
     this.isDirty = true;
   }
   
   private processRelationshipChange(change: RelationshipUpdate): void {
     // Trigger relationship animation
     if (change.operation === 'add') {
       // Animate connection between columns
       const sourceState = this.columnStates.get(change.source);
       const targetState = this.columnStates.get(change.target);
       
       if (sourceState && targetState) {
         // Create connection animation
         this.activationAnimator.animateSpreadingActivation(
           change.source,
           [change.target],
           100,
           change.weight
         );
       }
     }
   }
   
   private updateStats(updateTime: number): void {
     this.updateTimes.push(updateTime);
     if (this.updateTimes.length > 60) {
       this.updateTimes.shift();
     }
     
     this.stats.totalUpdates++;
     this.stats.lastUpdateTime = performance.now();
     this.stats.queueSize = this.pendingUpdates.length;
     this.stats.averageUpdateTime = this.updateTimes.reduce((a, b) => a + b, 0) / this.updateTimes.length;
     
     // Calculate updates per second
     const now = performance.now();
     if (now - this.lastStatsUpdate >= 1000) {
       this.stats.updatesPerSecond = this.updateTimes.length;
       this.lastStatsUpdate = now;
       
       if (this.onStatsUpdate) {
         this.onStatsUpdate({ ...this.stats });
       }
       
       if (this.config.debugMode) {
         console.log('Update Stats:', this.stats);
       }
     }
   }
   
   private updatePreviousStates(): void {
     for (const [id, state] of this.columnStates) {
       this.previousStates.set(id, { ...state });
     }
   }
   
   private arraysEqual<T>(a: T[], b: T[]): boolean {
     if (a.length !== b.length) return false;
     for (let i = 0; i < a.length; i++) {
       if (a[i] !== b[i]) return false;
     }
     return true;
   }
   
   // Public API
   public forceUpdate(): void {
     this.isDirty = true;
     this.performUpdate();
   }
   
   public forceRender(): void {
     this.isDirty = true;
   }
   
   public getStats(): UpdateStats {
     return { ...this.stats };
   }
   
   public getColumnState(columnId: number): ColumnState | undefined {
     return this.columnStates.get(columnId);
   }
   
   public getAllColumnStates(): ColumnState[] {
     return Array.from(this.columnStates.values());
   }
   
   public queueUpdate(update: SystemUpdate): void {
     this.pendingUpdates.push(update);
   }
   
   public setUpdateInterval(interval: number): void {
     this.config.updateInterval = interval;
     
     if (this.isRunning) {
       this.stop();
       this.start();
     }
   }
   
   public enableDeltaRendering(enabled: boolean): void {
     this.config.enableDeltaRendering = enabled;
   }
   
   public clearQueue(): void {
     this.pendingUpdates = [];
     this.updateQueue = [];
   }
   
   public dispose(): void {
     this.stop();
     this.clearQueue();
     this.columnStates.clear();
     this.previousStates.clear();
   }
   ```

## Expected Outputs
- Efficient real-time state synchronization between WASM and UI
- Delta rendering optimization reducing unnecessary redraws
- Change detection system tracking modifications automatically
- Performance monitoring with detailed statistics
- Event-driven update system with configurable intervals
- Smooth animation integration with state changes

## Validation
1. Updates maintain 60 FPS with 1000+ columns changing simultaneously
2. Delta rendering reduces rendering overhead by >70%
3. Change detection accurately identifies all state modifications
4. Memory usage remains stable during extended operation
5. Animation triggers correctly sync with state changes

## Next Steps
- Integration testing with complete Phase 9 WASM system
- Performance optimization and final debugging