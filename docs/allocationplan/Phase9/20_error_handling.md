# Micro-Phase 9.20: Comprehensive Error Handling Implementation

## Objective
Implement robust error handling and recovery mechanisms for all CortexKG operations with proper error classification, logging, and user-friendly error reporting.

## Prerequisites
- Completed micro-phase 9.19 (Promise handling)
- Async API wrapper functional
- Promise management system operational

## Task Description
Create a comprehensive error handling system that categorizes errors, provides meaningful error messages, implements automatic recovery strategies, and maintains detailed error logs for debugging and monitoring purposes.

## Specific Actions

1. **Create error classification system**:
   ```typescript
   // src/core/ErrorTypes.ts
   export enum ErrorCategory {
     INITIALIZATION = 'initialization',
     WASM_RUNTIME = 'wasm_runtime',
     MEMORY = 'memory',
     NETWORK = 'network',
     VALIDATION = 'validation',
     TIMEOUT = 'timeout',
     CANCELLED = 'cancelled',
     CONFIGURATION = 'configuration',
     PERFORMANCE = 'performance',
     UNKNOWN = 'unknown'
   }

   export enum ErrorSeverity {
     LOW = 'low',         // Non-critical, operation can continue
     MEDIUM = 'medium',   // Important but recoverable
     HIGH = 'high',       // Critical, affects functionality
     CRITICAL = 'critical' // System-level failure
   }

   export interface ErrorContext {
     operation: string;
     timestamp: Date;
     userAgent?: string;
     wasmVersion?: string;
     memoryUsage?: number;
     activeOperations?: number;
     stackTrace?: string;
     additionalData?: Record<string, any>;
   }

   export interface CortexError {
     id: string;
     category: ErrorCategory;
     severity: ErrorSeverity;
     message: string;
     originalError?: Error;
     context: ErrorContext;
     recoverable: boolean;
     retryable: boolean;
     userFriendlyMessage: string;
     troubleshootingSteps: string[];
   }

   export class CortexKGError extends Error {
     public readonly id: string;
     public readonly category: ErrorCategory;
     public readonly severity: ErrorSeverity;
     public readonly context: ErrorContext;
     public readonly recoverable: boolean;
     public readonly retryable: boolean;
     public readonly userFriendlyMessage: string;
     public readonly troubleshootingSteps: string[];

     constructor(cortexError: CortexError) {
       super(cortexError.message);
       this.name = 'CortexKGError';
       this.id = cortexError.id;
       this.category = cortexError.category;
       this.severity = cortexError.severity;
       this.context = cortexError.context;
       this.recoverable = cortexError.recoverable;
       this.retryable = cortexError.retryable;
       this.userFriendlyMessage = cortexError.userFriendlyMessage;
       this.troubleshootingSteps = cortexError.troubleshootingSteps;

       // Maintain proper stack trace
       if (Error.captureStackTrace) {
         Error.captureStackTrace(this, CortexKGError);
       }
     }
   }

   export interface ErrorRecoveryStrategy {
     category: ErrorCategory;
     canRecover: (error: CortexKGError) => boolean;
     recover: (error: CortexKGError) => Promise<boolean>;
     maxAttempts: number;
     backoffMs: number;
   }
   ```

2. **Create error factory and classifier**:
   ```typescript
   // src/core/ErrorFactory.ts
   import { ErrorCategory, ErrorSeverity, CortexError, CortexKGError, ErrorContext } from './ErrorTypes';

   export class ErrorFactory {
     private static generateId(): string {
       return `ctx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
     }

     private static createContext(operation: string, additionalData?: Record<string, any>): ErrorContext {
       return {
         operation,
         timestamp: new Date(),
         userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
         wasmVersion: '1.0.0', // TODO: Get from actual WASM module
         memoryUsage: typeof performance !== 'undefined' && performance.memory 
           ? performance.memory.usedJSHeapSize / (1024 * 1024) 
           : undefined,
         stackTrace: new Error().stack,
         additionalData
       };
     }

     static createInitializationError(originalError: Error, operation: string): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.INITIALIZATION,
         severity: ErrorSeverity.CRITICAL,
         message: `Initialization failed: ${originalError.message}`,
         originalError,
         context: this.createContext(operation),
         recoverable: true,
         retryable: true,
         userFriendlyMessage: 'Failed to initialize CortexKG. Please try refreshing the page.',
         troubleshootingSteps: [
           'Refresh the page',
           'Check internet connection',
           'Ensure WebAssembly is supported',
           'Try a different browser',
           'Contact support if the problem persists'
         ]
       };

       return new CortexKGError(error);
     }

     static createWasmRuntimeError(originalError: Error, operation: string): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.WASM_RUNTIME,
         severity: ErrorSeverity.HIGH,
         message: `WASM runtime error: ${originalError.message}`,
         originalError,
         context: this.createContext(operation),
         recoverable: false,
         retryable: false,
         userFriendlyMessage: 'An internal error occurred. The system needs to be restarted.',
         troubleshootingSteps: [
           'Refresh the page to restart the system',
           'Clear browser cache and cookies',
           'Try using an incognito/private window',
           'Update your browser to the latest version'
         ]
       };

       return new CortexKGError(error);
     }

     static createMemoryError(operation: string, memoryUsage?: number): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.MEMORY,
         severity: ErrorSeverity.HIGH,
         message: `Memory allocation failed during ${operation}`,
         context: this.createContext(operation, { memoryUsage }),
         recoverable: true,
         retryable: false,
         userFriendlyMessage: 'The system is running low on memory. Please free up some resources.',
         troubleshootingSteps: [
           'Close other browser tabs',
           'Refresh the page',
           'Free up system memory',
           'Try processing smaller batches of data'
         ]
       };

       return new CortexKGError(error);
     }

     static createTimeoutError(operation: string, timeoutMs: number): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.TIMEOUT,
         severity: ErrorSeverity.MEDIUM,
         message: `Operation "${operation}" timed out after ${timeoutMs}ms`,
         context: this.createContext(operation, { timeoutMs }),
         recoverable: true,
         retryable: true,
         userFriendlyMessage: 'The operation is taking longer than expected. Please try again.',
         troubleshootingSteps: [
           'Try again with a longer timeout',
           'Check your internet connection',
           'Try processing smaller amounts of data',
           'Contact support if timeouts persist'
         ]
       };

       return new CortexKGError(error);
     }

     static createValidationError(field: string, value: any, operation: string): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.VALIDATION,
         severity: ErrorSeverity.LOW,
         message: `Validation failed for ${field}: ${value}`,
         context: this.createContext(operation, { field, value }),
         recoverable: true,
         retryable: false,
         userFriendlyMessage: 'The provided input is not valid. Please check your data and try again.',
         troubleshootingSteps: [
           'Check the input format',
           'Ensure all required fields are provided',
           'Verify data types match expected format',
           'Consult the API documentation'
         ]
       };

       return new CortexKGError(error);
     }

     static createNetworkError(originalError: Error, operation: string): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.NETWORK,
         severity: ErrorSeverity.MEDIUM,
         message: `Network error during ${operation}: ${originalError.message}`,
         originalError,
         context: this.createContext(operation),
         recoverable: true,
         retryable: true,
         userFriendlyMessage: 'Network connection issue. Please check your internet connection and try again.',
         troubleshootingSteps: [
           'Check internet connection',
           'Try again in a few moments',
           'Disable VPN if active',
           'Contact your network administrator if problems persist'
         ]
       };

       return new CortexKGError(error);
     }

     static createCancellationError(operation: string): CortexKGError {
       const error: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.CANCELLED,
         severity: ErrorSeverity.LOW,
         message: `Operation "${operation}" was cancelled by user`,
         context: this.createContext(operation),
         recoverable: true,
         retryable: true,
         userFriendlyMessage: 'Operation was cancelled.',
         troubleshootingSteps: [
           'Start the operation again if needed'
         ]
       };

       return new CortexKGError(error);
     }

     static classifyError(error: Error, operation: string): CortexKGError {
       if (error instanceof CortexKGError) {
         return error;
       }

       // Pattern matching for common error types
       const message = error.message.toLowerCase();

       if (message.includes('timeout') || message.includes('timed out')) {
         return this.createTimeoutError(operation, 30000);
       }

       if (message.includes('memory') || message.includes('allocation')) {
         return this.createMemoryError(operation);
       }

       if (message.includes('network') || message.includes('fetch')) {
         return this.createNetworkError(error, operation);
       }

       if (message.includes('abort') || message.includes('cancel')) {
         return this.createCancellationError(operation);
       }

       if (message.includes('wasm') || message.includes('webassembly')) {
         return this.createWasmRuntimeError(error, operation);
       }

       // Default to unknown error
       const unknownError: CortexError = {
         id: this.generateId(),
         category: ErrorCategory.UNKNOWN,
         severity: ErrorSeverity.MEDIUM,
         message: `Unknown error in ${operation}: ${error.message}`,
         originalError: error,
         context: this.createContext(operation),
         recoverable: true,
         retryable: true,
         userFriendlyMessage: 'An unexpected error occurred. Please try again.',
         troubleshootingSteps: [
           'Try the operation again',
           'Refresh the page if the problem persists',
           'Contact support with error details'
         ]
       };

       return new CortexKGError(unknownError);
     }
   }
   ```

3. **Create error recovery manager**:
   ```typescript
   // src/core/ErrorRecoveryManager.ts
   import { CortexKGError, ErrorCategory, ErrorRecoveryStrategy } from './ErrorTypes';
   import { ErrorLogger } from './ErrorLogger';

   export interface RecoveryResult {
     success: boolean;
     attemptCount: number;
     totalTimeMs: number;
     strategy?: string;
     finalError?: CortexKGError;
   }

   export class ErrorRecoveryManager {
     private strategies: Map<ErrorCategory, ErrorRecoveryStrategy[]> = new Map();
     private logger: ErrorLogger;
     private maxGlobalRetries: number = 3;

     constructor(logger: ErrorLogger) {
       this.logger = logger;
       this.initializeDefaultStrategies();
     }

     private initializeDefaultStrategies(): void {
       // Initialization error recovery
       this.addStrategy({
         category: ErrorCategory.INITIALIZATION,
         canRecover: (error) => error.retryable,
         recover: async (error) => {
           // Clear any cached WASM modules and try again
           if (typeof caches !== 'undefined') {
             const cacheNames = await caches.keys();
             await Promise.all(
               cacheNames.map(name => 
                 name.includes('wasm') ? caches.delete(name) : Promise.resolve()
               )
             );
           }
           return true;
         },
         maxAttempts: 3,
         backoffMs: 1000
       });

       // Memory error recovery
       this.addStrategy({
         category: ErrorCategory.MEMORY,
         canRecover: (error) => error.recoverable,
         recover: async (error) => {
           // Force garbage collection if available
           if (typeof window !== 'undefined' && (window as any).gc) {
             (window as any).gc();
           }
           
           // Wait a bit for memory to free up
           await new Promise(resolve => setTimeout(resolve, 500));
           return true;
         },
         maxAttempts: 2,
         backoffMs: 500
       });

       // Network error recovery
       this.addStrategy({
         category: ErrorCategory.NETWORK,
         canRecover: (error) => error.retryable,
         recover: async (error) => {
           // Simple exponential backoff for network issues
           const backoff = Math.min(1000 * Math.pow(2, 3), 10000);
           await new Promise(resolve => setTimeout(resolve, backoff));
           return true;
         },
         maxAttempts: 5,
         backoffMs: 1000
       });

       // Timeout error recovery
       this.addStrategy({
         category: ErrorCategory.TIMEOUT,
         canRecover: (error) => error.retryable,
         recover: async (error) => {
           // No special recovery needed, just retry with longer timeout
           return true;
         },
         maxAttempts: 2,
         backoffMs: 0
       });
     }

     addStrategy(strategy: ErrorRecoveryStrategy): void {
       if (!this.strategies.has(strategy.category)) {
         this.strategies.set(strategy.category, []);
       }
       this.strategies.get(strategy.category)!.push(strategy);
     }

     async attemptRecovery<T>(
       error: CortexKGError,
       operation: () => Promise<T>,
       maxRetries?: number
     ): Promise<{ result?: T; recovery: RecoveryResult }> {
       const startTime = performance.now();
       const maxAttempts = maxRetries || this.maxGlobalRetries;
       let lastError = error;
       let attemptCount = 0;

       this.logger.logError(error);

       const strategies = this.strategies.get(error.category) || [];
       
       for (const strategy of strategies) {
         if (!strategy.canRecover(error)) {
           continue;
         }

         for (let attempt = 0; attempt < strategy.maxAttempts && attemptCount < maxAttempts; attempt++) {
           attemptCount++;

           try {
             // Attempt recovery
             const recoverySuccess = await strategy.recover(error);
             if (!recoverySuccess) {
               continue;
             }

             // Wait for backoff period
             if (strategy.backoffMs > 0) {
               await new Promise(resolve => 
                 setTimeout(resolve, strategy.backoffMs * (attempt + 1))
               );
             }

             // Retry the original operation
             const result = await operation();

             // Success!
             const recovery: RecoveryResult = {
               success: true,
               attemptCount,
               totalTimeMs: performance.now() - startTime,
               strategy: strategy.category
             };

             this.logger.logRecovery(error, recovery);
             return { result, recovery };

           } catch (retryError) {
             lastError = retryError instanceof CortexKGError 
               ? retryError 
               : ErrorFactory.classifyError(retryError as Error, 'recovery_retry');

             this.logger.logError(lastError);
           }
         }
       }

       // All recovery attempts failed
       const recovery: RecoveryResult = {
         success: false,
         attemptCount,
         totalTimeMs: performance.now() - startTime,
         finalError: lastError
       };

       this.logger.logRecoveryFailure(error, recovery);
       return { recovery };
     }

     getRecoveryStrategies(category: ErrorCategory): ErrorRecoveryStrategy[] {
       return this.strategies.get(category) || [];
     }

     clearStrategies(category?: ErrorCategory): void {
       if (category) {
         this.strategies.delete(category);
       } else {
         this.strategies.clear();
         this.initializeDefaultStrategies();
       }
     }
   }
   ```

4. **Create error logging system**:
   ```typescript
   // src/core/ErrorLogger.ts
   import { CortexKGError, ErrorSeverity } from './ErrorTypes';
   import { RecoveryResult } from './ErrorRecoveryManager';

   export interface ErrorLogEntry {
     error: CortexKGError;
     timestamp: Date;
     sessionId: string;
     url: string;
   }

   export interface RecoveryLogEntry {
     originalError: CortexKGError;
     recovery: RecoveryResult;
     timestamp: Date;
     sessionId: string;
   }

   export interface ErrorStats {
     totalErrors: number;
     errorsByCategory: Record<string, number>;
     errorsBySeverity: Record<string, number>;
     recoveryRate: number;
     avgRecoveryTime: number;
   }

   export class ErrorLogger {
     private errorLog: ErrorLogEntry[] = [];
     private recoveryLog: RecoveryLogEntry[] = [];
     private sessionId: string;
     private maxLogSize: number = 1000;
     private enableConsoleLogging: boolean = true;
     private enableRemoteLogging: boolean = false;

     constructor(options: {
       maxLogSize?: number;
       enableConsoleLogging?: boolean;
       enableRemoteLogging?: boolean;
     } = {}) {
       this.maxLogSize = options.maxLogSize || 1000;
       this.enableConsoleLogging = options.enableConsoleLogging ?? true;
       this.enableRemoteLogging = options.enableRemoteLogging ?? false;
       this.sessionId = this.generateSessionId();
     }

     private generateSessionId(): string {
       return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
     }

     logError(error: CortexKGError): void {
       const entry: ErrorLogEntry = {
         error,
         timestamp: new Date(),
         sessionId: this.sessionId,
         url: typeof window !== 'undefined' ? window.location.href : 'unknown'
       };

       this.errorLog.push(entry);
       this.trimLog(this.errorLog);

       if (this.enableConsoleLogging) {
         this.logToConsole(error);
       }

       if (this.enableRemoteLogging) {
         this.logToRemote(entry).catch(console.error);
       }
     }

     logRecovery(originalError: CortexKGError, recovery: RecoveryResult): void {
       const entry: RecoveryLogEntry = {
         originalError,
         recovery,
         timestamp: new Date(),
         sessionId: this.sessionId
       };

       this.recoveryLog.push(entry);
       this.trimLog(this.recoveryLog);

       if (this.enableConsoleLogging) {
         console.log(`🔄 Recovery successful for ${originalError.id}:`, {
           strategy: recovery.strategy,
           attempts: recovery.attemptCount,
           timeMs: recovery.totalTimeMs
         });
       }
     }

     logRecoveryFailure(originalError: CortexKGError, recovery: RecoveryResult): void {
       const entry: RecoveryLogEntry = {
         originalError,
         recovery,
         timestamp: new Date(),
         sessionId: this.sessionId
       };

       this.recoveryLog.push(entry);
       this.trimLog(this.recoveryLog);

       if (this.enableConsoleLogging) {
         console.error(`❌ Recovery failed for ${originalError.id}:`, {
           attempts: recovery.attemptCount,
           timeMs: recovery.totalTimeMs,
           finalError: recovery.finalError
         });
       }
     }

     private logToConsole(error: CortexKGError): void {
       const emoji = this.getSeverityEmoji(error.severity);
       const style = this.getSeverityStyle(error.severity);

       console.group(`${emoji} CortexKG Error [${error.category}]`);
       console.error(`%c${error.userFriendlyMessage}`, style);
       console.log('Error ID:', error.id);
       console.log('Message:', error.message);
       console.log('Context:', error.context);
       console.log('Troubleshooting:', error.troubleshootingSteps);
       if (error.originalError) {
         console.log('Original error:', error.originalError);
       }
       console.groupEnd();
     }

     private getSeverityEmoji(severity: ErrorSeverity): string {
       switch (severity) {
         case ErrorSeverity.LOW: return '⚠️';
         case ErrorSeverity.MEDIUM: return '🟡';
         case ErrorSeverity.HIGH: return '🔴';
         case ErrorSeverity.CRITICAL: return '💥';
         default: return '❓';
       }
     }

     private getSeverityStyle(severity: ErrorSeverity): string {
       switch (severity) {
         case ErrorSeverity.LOW: return 'color: orange; font-weight: bold;';
         case ErrorSeverity.MEDIUM: return 'color: gold; font-weight: bold;';
         case ErrorSeverity.HIGH: return 'color: red; font-weight: bold;';
         case ErrorSeverity.CRITICAL: return 'color: darkred; font-weight: bold; font-size: 1.2em;';
         default: return 'color: gray; font-weight: bold;';
       }
     }

     private async logToRemote(entry: ErrorLogEntry): Promise<void> {
       try {
         // In a real implementation, this would send to your error tracking service
         await fetch('/api/errors', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify(entry)
         });
       } catch (error) {
         // Don't throw - we don't want error logging to break the app
         console.warn('Failed to log error remotely:', error);
       }
     }

     private trimLog<T>(log: T[]): void {
       if (log.length > this.maxLogSize) {
         log.splice(0, log.length - this.maxLogSize);
       }
     }

     getErrorStats(): ErrorStats {
       const totalErrors = this.errorLog.length;
       const successfulRecoveries = this.recoveryLog.filter(r => r.recovery.success).length;
       const totalRecoveryAttempts = this.recoveryLog.length;

       const errorsByCategory: Record<string, number> = {};
       const errorsBySeverity: Record<string, number> = {};

       for (const entry of this.errorLog) {
         errorsByCategory[entry.error.category] = (errorsByCategory[entry.error.category] || 0) + 1;
         errorsBySeverity[entry.error.severity] = (errorsBySeverity[entry.error.severity] || 0) + 1;
       }

       const avgRecoveryTime = totalRecoveryAttempts > 0
         ? this.recoveryLog.reduce((sum, r) => sum + r.recovery.totalTimeMs, 0) / totalRecoveryAttempts
         : 0;

       return {
         totalErrors,
         errorsByCategory,
         errorsBySeverity,
         recoveryRate: totalRecoveryAttempts > 0 ? successfulRecoveries / totalRecoveryAttempts : 0,
         avgRecoveryTime
       };
     }

     exportLogs(): { errors: ErrorLogEntry[]; recoveries: RecoveryLogEntry[] } {
       return {
         errors: [...this.errorLog],
         recoveries: [...this.recoveryLog]
       };
     }

     clearLogs(): void {
       this.errorLog.length = 0;
       this.recoveryLog.length = 0;
     }
   }
   ```

## Expected Outputs
- Comprehensive error classification and handling system
- Automatic error recovery with configurable strategies
- Detailed error logging with console and remote capabilities
- User-friendly error messages with troubleshooting steps
- Error statistics and monitoring capabilities

## Validation
1. All error types are properly classified and handled
2. Recovery strategies successfully handle common failure scenarios
3. Error messages are user-friendly and actionable
4. Logging system captures all necessary debugging information
5. Error statistics provide useful insights for monitoring

## Next Steps
- Create HTML structure for web interface (micro-phase 9.21)
- Implement responsive CSS design (micro-phase 9.22)