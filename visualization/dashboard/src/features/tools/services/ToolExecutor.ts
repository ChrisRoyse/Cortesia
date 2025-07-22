import { JSONSchema7 } from 'json-schema';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { v4 as uuidv4 } from 'uuid';
import { Observable, Subject } from 'rxjs';
import { MCPTool, ToolExecution, ExecutionStatus } from '../types';

export interface ValidationResult {
  valid: boolean;
  errors?: Array<{
    path: string;
    message: string;
    keyword: string;
    params: any;
  }>;
}

export interface ExecutionUpdate {
  executionId: string;
  type: 'progress' | 'log' | 'stream' | 'status';
  data: any;
  timestamp: number;
}

export class ToolExecutor {
  private ajv: Ajv;
  private executions: Map<string, ToolExecution>;
  private executionStreams: Map<string, Subject<ExecutionUpdate>>;
  private abortControllers: Map<string, AbortController>;

  constructor() {
    this.ajv = new Ajv({ allErrors: true, verbose: true });
    addFormats(this.ajv);
    this.executions = new Map();
    this.executionStreams = new Map();
    this.abortControllers = new Map();
  }

  /**
   * Validates input against a JSON schema
   */
  validateInput(schema: JSONSchema7, input: any): ValidationResult {
    const validate = this.ajv.compile(schema);
    const valid = validate(input);

    if (!valid && validate.errors) {
      return {
        valid: false,
        errors: validate.errors.map(error => ({
          path: error.instancePath || 'root',
          message: error.message || 'Validation error',
          keyword: error.keyword,
          params: error.params,
        })),
      };
    }

    return { valid: true };
  }

  /**
   * Executes a tool with the given input
   */
  async executeTool(tool: MCPTool, input: any): Promise<ToolExecution> {
    const executionId = uuidv4();
    const startTime = Date.now();

    // Create execution record
    const execution: ToolExecution = {
      id: executionId,
      toolId: tool.id,
      input,
      output: null,
      error: null,
      status: 'running',
      startTime,
      endTime: null,
      metadata: {
        toolName: tool.name,
        toolVersion: tool.version,
      },
    };

    this.executions.set(executionId, execution);

    // Create stream for updates
    const stream = new Subject<ExecutionUpdate>();
    this.executionStreams.set(executionId, stream);

    // Create abort controller
    const abortController = new AbortController();
    this.abortControllers.set(executionId, abortController);

    try {
      // Validate input
      const validation = this.validateInput(tool.inputSchema, input);
      if (!validation.valid) {
        throw new Error(`Input validation failed: ${JSON.stringify(validation.errors)}`);
      }

      // Send progress update
      this.sendUpdate(executionId, 'progress', { message: 'Validating input...', progress: 10 });

      // Execute tool via MCP protocol
      const result = await this.executeViaMCP(tool, input, executionId, abortController.signal);

      // Update execution record
      execution.output = result;
      execution.status = 'success';
      execution.endTime = Date.now();

      // Send completion update
      this.sendUpdate(executionId, 'status', { status: 'success' });

    } catch (error: any) {
      // Handle cancellation
      if (error.name === 'AbortError') {
        execution.status = 'cancelled';
        execution.error = 'Execution cancelled by user';
        this.sendUpdate(executionId, 'status', { status: 'cancelled' });
      } else {
        // Handle other errors
        execution.status = 'error';
        execution.error = error.message || 'Unknown error occurred';
        this.sendUpdate(executionId, 'status', { status: 'error', error: execution.error });
      }
      execution.endTime = Date.now();
    } finally {
      // Clean up
      this.abortControllers.delete(executionId);
      
      // Complete stream after a delay to ensure all updates are received
      setTimeout(() => {
        stream.complete();
        this.executionStreams.delete(executionId);
      }, 1000);
    }

    this.executions.set(executionId, execution);
    return execution;
  }

  /**
   * Executes tool via MCP protocol
   */
  private async executeViaMCP(
    tool: MCPTool,
    input: any,
    executionId: string,
    signal: AbortSignal
  ): Promise<any> {
    // Send execution start update
    this.sendUpdate(executionId, 'progress', { message: 'Connecting to tool...', progress: 20 });

    // Simulate MCP protocol execution
    // In a real implementation, this would communicate with the MCP server
    return new Promise((resolve, reject) => {
      // Check if already aborted
      if (signal.aborted) {
        reject(new Error('AbortError'));
        return;
      }

      // Listen for abort
      signal.addEventListener('abort', () => {
        reject(new Error('AbortError'));
      });

      // Simulate async execution with progress updates
      let progress = 30;
      const progressInterval = setInterval(() => {
        if (signal.aborted) {
          clearInterval(progressInterval);
          return;
        }

        progress = Math.min(progress + 10, 90);
        this.sendUpdate(executionId, 'progress', {
          message: 'Processing...',
          progress,
        });
      }, 500);

      // Simulate execution time
      setTimeout(() => {
        clearInterval(progressInterval);

        if (signal.aborted) {
          reject(new Error('AbortError'));
          return;
        }

        // Send final progress
        this.sendUpdate(executionId, 'progress', {
          message: 'Finalizing...',
          progress: 100,
        });

        // Simulate different types of responses based on tool
        const mockResponse = this.generateMockResponse(tool, input);
        resolve(mockResponse);
      }, 2000 + Math.random() * 3000); // 2-5 seconds
    });
  }

  /**
   * Generates mock response for testing
   */
  private generateMockResponse(tool: MCPTool, input: any): any {
    // LLMKG-specific mock responses
    if (tool.name.includes('query')) {
      return {
        results: [
          { entity: 'Node1', relation: 'connects_to', target: 'Node2', weight: 0.8 },
          { entity: 'Node2', relation: 'activates', target: 'Node3', weight: 0.6 },
        ],
        count: 2,
        executionTime: Math.random() * 100,
      };
    }

    if (tool.name.includes('cognitive')) {
      return {
        pattern: 'hierarchical',
        activation: Array.from({ length: 10 }, () => Math.random()),
        timestamp: Date.now(),
      };
    }

    if (tool.name.includes('memory')) {
      return {
        stored: true,
        id: uuidv4(),
        size: Math.floor(Math.random() * 1000),
      };
    }

    // Default response
    return {
      success: true,
      input: input,
      processedAt: new Date().toISOString(),
      metadata: {
        toolName: tool.name,
        toolVersion: tool.version,
      },
    };
  }

  /**
   * Cancels an ongoing execution
   */
  cancelExecution(executionId: string): void {
    const controller = this.abortControllers.get(executionId);
    if (controller) {
      controller.abort();
    }
  }

  /**
   * Gets the execution stream for real-time updates
   */
  getExecutionStream(executionId: string): Observable<ExecutionUpdate> {
    const stream = this.executionStreams.get(executionId);
    if (!stream) {
      throw new Error(`No execution stream found for ${executionId}`);
    }
    return stream.asObservable();
  }

  /**
   * Gets execution by ID
   */
  getExecution(executionId: string): ToolExecution | undefined {
    return this.executions.get(executionId);
  }

  /**
   * Sends an update to the execution stream
   */
  private sendUpdate(executionId: string, type: ExecutionUpdate['type'], data: any): void {
    const stream = this.executionStreams.get(executionId);
    if (stream) {
      stream.next({
        executionId,
        type,
        data,
        timestamp: Date.now(),
      });
    }
  }

  /**
   * Clears all executions and streams
   */
  clear(): void {
    // Cancel all ongoing executions
    this.abortControllers.forEach(controller => controller.abort());
    
    // Complete all streams
    this.executionStreams.forEach(stream => stream.complete());
    
    // Clear maps
    this.executions.clear();
    this.executionStreams.clear();
    this.abortControllers.clear();
  }
}