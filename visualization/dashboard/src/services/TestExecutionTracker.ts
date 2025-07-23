import { EventEmitter } from 'events';
import { TestDiscoveryService, TestExecutionSummary, CargoTestResult, TestModule } from './TestDiscoveryService';

export interface TestExecution {
  id: string;
  startTime: Date;
  endTime?: Date;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  testPattern?: string;
  category?: string;
  summary?: TestExecutionSummary;
  progress: {
    current: number;
    total: number;
    currentTest?: string;
  };
  logs: TestExecutionLog[];
}

export interface TestExecutionLog {
  timestamp: Date;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  testName?: string;
}

export interface TestSuiteDefinition {
  id: string;
  name: string;
  description: string;
  category: string;
  testPattern: string;
  modules: TestModule[];
  totalTests: number;
  tags: string[];
  enabled: boolean;
}

export interface TestExecutionHistory {
  executions: TestExecution[];
  totalRuns: number;
  successRate: number;
  averageExecutionTime: number;
  lastRun?: Date;
}

export class TestExecutionTracker extends EventEmitter {
  private testDiscovery: TestDiscoveryService;
  private activeExecutions: Map<string, TestExecution> = new Map();
  private executionHistory: TestExecution[] = [];
  private maxHistorySize = 100;
  
  constructor(projectRoot?: string) {
    super();
    this.testDiscovery = new TestDiscoveryService(projectRoot);
  }

  /**
   * Get available test suites based on discovered tests
   */
  async getTestSuites(): Promise<TestSuiteDefinition[]> {
    // First try to fetch from backend API
    try {
      const response = await fetch('http://localhost:8082/api/tests/discover');
      if (response.ok) {
        const data = await response.json();
        const suites: TestSuiteDefinition[] = [];
        
        // Transform backend response to TestSuiteDefinition format
        for (const suite of data.suites) {
          suites.push({
            id: `suite-${suite.name.toLowerCase().replace(/[^a-z0-9]/g, '-')}`,
            name: suite.name,
            description: suite.description || `Tests for ${suite.name}`,
            category: suite.test_type || 'Unit',
            testPattern: suite.name,
            modules: [],
            totalTests: suite.test_count || 0,
            tags: suite.tags || [],
            enabled: true
          });
        }
        
        return suites;
      }
    } catch (error) {
      console.warn('Failed to fetch test suites from backend, falling back to local discovery:', error);
    }
    
    // Fallback to local discovery
    const inventory = await this.testDiscovery.discoverTests();
    const suites: TestSuiteDefinition[] = [];
    
    // Create suites by category
    for (const [category, modules] of Object.entries(inventory.categories)) {
      const totalTests = modules.reduce((sum, m) => sum + m.totalTests, 0);
      
      suites.push({
        id: `suite-${category.toLowerCase().replace(/\s+/g, '-')}`,
        name: `${category} Tests`,
        description: `All tests in the ${category} category`,
        category,
        testPattern: this.createTestPattern(modules),
        modules,
        totalTests,
        tags: [category, 'auto-generated'],
        enabled: true
      });
    }
    
    // Create module-specific suites for larger modules
    for (const module of inventory.modules) {
      if (module.totalTests >= 5) {
        suites.push({
          id: `module-${module.relativePath.replace(/[\/\\]/g, '-').replace('.rs', '')}`,
          name: `${this.getModuleDisplayName(module)} Tests`,
          description: `Tests for ${module.relativePath}`,
          category: module.category,
          testPattern: this.createModuleTestPattern(module),
          modules: [module],
          totalTests: module.totalTests,
          tags: [module.category, 'module', module.moduleType],
          enabled: true
        });
      }
    }
    
    // Add special suites
    suites.push(
      {
        id: 'all-tests',
        name: 'All Tests',
        description: 'Run all available tests in the project',
        category: 'Complete',
        testPattern: '',
        modules: inventory.modules,
        totalTests: inventory.totalTests,
        tags: ['complete', 'all'],
        enabled: true
      },
      {
        id: 'quick-tests',
        name: 'Quick Tests',
        description: 'Fast unit tests for quick feedback',
        category: 'Quick',
        testPattern: 'test_ --lib',
        modules: inventory.modules.filter(m => m.moduleType === 'unit'),
        totalTests: inventory.modules.filter(m => m.moduleType === 'unit').reduce((sum, m) => sum + m.totalTests, 0),
        tags: ['quick', 'unit'],
        enabled: true
      },
      {
        id: 'integration-tests',
        name: 'Integration Tests',
        description: 'Full integration test suite',
        category: 'Integration',
        testPattern: '--test "*"',
        modules: inventory.modules.filter(m => m.moduleType === 'integration'),
        totalTests: inventory.modules.filter(m => m.moduleType === 'integration').reduce((sum, m) => sum + m.totalTests, 0),
        tags: ['integration', 'slow'],
        enabled: true
      }
    );
    
    return suites.filter(suite => suite.totalTests > 0);
  }

  /**
   * Execute a test suite
   */
  async executeTestSuite(
    suiteId: string, 
    options: {
      release?: boolean;
      nocapture?: boolean;
      ignored?: boolean;
      features?: string[];
    } = {}
  ): Promise<string> {
    const suites = await this.getTestSuites();
    const suite = suites.find(s => s.id === suiteId);
    
    if (!suite) {
      throw new Error(`Test suite ${suiteId} not found`);
    }
    
    // Call the backend API to execute tests
    try {
      const response = await fetch('http://localhost:8082/api/tests/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          suite_name: suite.name,
          filter: suite.testPattern,
          nocapture: options.nocapture || false,
          parallel: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const executionId = data.execution_id;
      
      // Create local execution tracking
      const execution: TestExecution = {
        id: executionId,
        startTime: new Date(),
        status: 'running',
        testPattern: suite.testPattern,
        category: suite.category,
        progress: {
          current: 0,
          total: suite.totalTests
        },
        logs: []
      };
      
      this.activeExecutions.set(executionId, execution);
      this.addLog(execution, 'info', `Starting test suite: ${suite.name}`);
      this.emit('executionStarted', execution);
      
      // The WebSocket connection will handle progress updates
      // We'll rely on the WebSocket service to update the execution status
      
      return executionId;
      
    } catch (error) {
      console.error('Failed to execute test suite:', error);
      
      // Create a local execution for error tracking
      const executionId = `exec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const execution: TestExecution = {
        id: executionId,
        startTime: new Date(),
        endTime: new Date(),
        status: 'failed',
        testPattern: suite.testPattern,
        category: suite.category,
        progress: {
          current: 0,
          total: suite.totalTests
        },
        logs: []
      };
      
      this.activeExecutions.set(executionId, execution);
      this.addLog(execution, 'error', `Failed to start test execution: ${error}`);
      this.emit('executionFailed', execution, error);
      
      // Move to history
      this.executionHistory.unshift(execution);
      if (this.executionHistory.length > this.maxHistorySize) {
        this.executionHistory = this.executionHistory.slice(0, this.maxHistorySize);
      }
      
      this.activeExecutions.delete(executionId);
      
      throw error;
    }
  }

  /**
   * Execute a specific test function
   */
  async executeTest(
    testName: string,
    options: {
      release?: boolean;
      nocapture?: boolean;
    } = {}
  ): Promise<string> {
    const executionId = `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const execution: TestExecution = {
      id: executionId,
      startTime: new Date(),
      status: 'running',
      testPattern: testName,
      progress: {
        current: 0,
        total: 1,
        currentTest: testName
      },
      logs: []
    };
    
    this.activeExecutions.set(executionId, execution);
    this.addLog(execution, 'info', `Starting test: ${testName}`);
    this.emit('executionStarted', execution);
    
    try {
      const summary = await this.testDiscovery.executeTests(testName, options);
      
      execution.endTime = new Date();
      execution.status = summary.failed > 0 ? 'failed' : 'completed';
      execution.summary = summary;
      execution.progress.current = 1;
      
      this.addLog(execution, 'info', `Test completed: ${summary.results[0]?.outcome || 'unknown'}`);
      this.emit('executionCompleted', execution);
      
    } catch (error) {
      execution.endTime = new Date();
      execution.status = 'failed';
      this.addLog(execution, 'error', `Test execution failed: ${error}`);
      this.emit('executionFailed', execution, error);
    } finally {
      this.executionHistory.unshift(execution);
      if (this.executionHistory.length > this.maxHistorySize) {
        this.executionHistory = this.executionHistory.slice(0, this.maxHistorySize);
      }
      
      this.activeExecutions.delete(executionId);
    }
    
    return executionId;
  }

  /**
   * Cancel a running test execution
   */
  async cancelExecution(executionId: string): Promise<boolean> {
    const execution = this.activeExecutions.get(executionId);
    if (!execution) {
      return false;
    }
    
    execution.status = 'cancelled';
    execution.endTime = new Date();
    this.addLog(execution, 'warning', 'Test execution cancelled by user');
    
    this.emit('executionCancelled', execution);
    this.activeExecutions.delete(executionId);
    
    return true;
  }

  /**
   * Get active executions
   */
  getActiveExecutions(): TestExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  /**
   * Get execution history
   */
  getExecutionHistory(): TestExecutionHistory {
    const totalRuns = this.executionHistory.length;
    const successfulRuns = this.executionHistory.filter(e => e.status === 'completed').length;
    const successRate = totalRuns > 0 ? (successfulRuns / totalRuns) * 100 : 0;
    
    const executionTimes = this.executionHistory
      .filter(e => e.endTime && e.startTime)
      .map(e => e.endTime!.getTime() - e.startTime.getTime());
    
    const averageExecutionTime = executionTimes.length > 0 
      ? executionTimes.reduce((sum, time) => sum + time, 0) / executionTimes.length
      : 0;
    
    return {
      executions: this.executionHistory,
      totalRuns,
      successRate,
      averageExecutionTime,
      lastRun: this.executionHistory[0]?.startTime
    };
  }

  /**
   * Get execution by ID
   */
  getExecution(executionId: string): TestExecution | undefined {
    return this.activeExecutions.get(executionId) || 
           this.executionHistory.find(e => e.id === executionId);
  }

  /**
   * Get test statistics
   */
  async getTestStatistics(): Promise<any> {
    const inventory = await this.testDiscovery.discoverTests();
    const history = this.getExecutionHistory();
    
    return {
      totalTests: inventory.totalTests,
      moduleCount: inventory.modules.length,
      categoryBreakdown: Object.entries(inventory.categories).map(([category, modules]) => ({
        category,
        testCount: modules.reduce((sum, m) => sum + m.totalTests, 0),
        moduleCount: modules.length
      })),
      executionHistory: {
        totalRuns: history.totalRuns,
        successRate: history.successRate,
        averageExecutionTime: history.averageExecutionTime,
        lastRun: history.lastRun
      },
      recentTrends: this.calculateTestTrends()
    };
  }

  /**
   * Start watching for test file changes
   */
  startWatching(): () => void {
    return this.testDiscovery.watchTestFiles((changedFile) => {
      this.emit('testFileChanged', changedFile);
    });
  }

  private addLog(execution: TestExecution, level: TestExecutionLog['level'], message: string, testName?: string) {
    execution.logs.push({
      timestamp: new Date(),
      level,
      message,
      testName
    });
    
    this.emit('executionLog', execution.id, {
      timestamp: new Date(),
      level,
      message,
      testName
    });
  }

  private createTestPattern(modules: TestModule[]): string {
    // Create a pattern that matches tests in these modules
    const paths = modules.map(m => m.relativePath.replace('.rs', '').replace(/[\/\\]/g, '::'));
    if (paths.length === 1) {
      return paths[0];
    }
    return ''; // Run all tests when multiple modules
  }

  private createModuleTestPattern(module: TestModule): string {
    return module.relativePath.replace('.rs', '').replace(/[\/\\]/g, '::');
  }

  private getModuleDisplayName(module: TestModule): string {
    const pathParts = module.relativePath.split(/[\/\\]/);
    const filename = pathParts[pathParts.length - 1].replace('.rs', '');
    
    // Convert snake_case to Title Case
    return filename.split('_').map(part => 
      part.charAt(0).toUpperCase() + part.slice(1)
    ).join(' ');
  }

  private calculateTestTrends(): any {
    const recentExecutions = this.executionHistory.slice(0, 10);
    if (recentExecutions.length < 2) {
      return { trend: 'stable', change: 0 };
    }
    
    const recent = recentExecutions.slice(0, 5);
    const older = recentExecutions.slice(5, 10);
    
    const recentSuccessRate = recent.filter(e => e.status === 'completed').length / recent.length;
    const olderSuccessRate = older.filter(e => e.status === 'completed').length / older.length;
    
    const change = recentSuccessRate - olderSuccessRate;
    
    return {
      trend: change > 0.1 ? 'improving' : change < -0.1 ? 'declining' : 'stable',
      change: Math.round(change * 100)
    };
  }
}

export const testExecutionTracker = new TestExecutionTracker();