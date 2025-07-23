import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);

export interface TestFunction {
  name: string;
  line: number;
  module: string;
  isIgnored: boolean;
  attributes: string[];
  description?: string;
}

export interface TestModule {
  filePath: string;
  relativePath: string;
  moduleType: 'unit' | 'integration' | 'benchmark' | 'example';
  testFunctions: TestFunction[];
  totalTests: number;
  category: string;
  lastModified: Date;
}

export interface TestSuiteInventory {
  modules: TestModule[];
  categories: { [key: string]: TestModule[] };
  totalTests: number;
  discoveredAt: Date;
}

export interface CargoTestResult {
  testName: string;
  outcome: 'passed' | 'failed' | 'ignored' | 'timeout';
  executionTime: number;
  stdout?: string;
  stderr?: string;
  failureMessage?: string;
}

export interface TestExecutionSummary {
  passed: number;
  failed: number;
  ignored: number;
  total: number;
  executionTime: number;
  results: CargoTestResult[];
}

export class TestDiscoveryService {
  private projectRoot: string;
  
  constructor(projectRoot: string = process.cwd()) {
    this.projectRoot = projectRoot;
  }

  /**
   * Discover all test files and functions in the LLMKG project
   */
  async discoverTests(): Promise<TestSuiteInventory> {
    const modules: TestModule[] = [];
    
    // Find all Rust files with tests
    const testFiles = await this.findTestFiles();
    
    for (const filePath of testFiles) {
      try {
        const module = await this.analyzeTestFile(filePath);
        if (module && module.totalTests > 0) {
          modules.push(module);
        }
      } catch (error) {
        console.warn(`Failed to analyze test file ${filePath}:`, error);
      }
    }

    // Categorize modules
    const categories = this.categorizeModules(modules);
    
    return {
      modules,
      categories,
      totalTests: modules.reduce((sum, m) => sum + m.totalTests, 0),
      discoveredAt: new Date()
    };
  }

  /**
   * Find all Rust files that contain tests
   */
  private async findTestFiles(): Promise<string[]> {
    const testFiles: string[] = [];
    
    // Search in src/ directory for inline tests
    const srcFiles = await this.findRustFiles(path.join(this.projectRoot, 'src'));
    for (const file of srcFiles) {
      if (await this.hasTestFunctions(file)) {
        testFiles.push(file);
      }
    }
    
    // Search in tests/ directory for integration tests
    const testsDir = path.join(this.projectRoot, 'tests');
    if (fs.existsSync(testsDir)) {
      const integrationTests = await this.findRustFiles(testsDir);
      testFiles.push(...integrationTests);
    }
    
    return testFiles;
  }

  /**
   * Recursively find all .rs files in a directory
   */
  private async findRustFiles(dir: string): Promise<string[]> {
    const files: string[] = [];
    
    if (!fs.existsSync(dir)) {
      return files;
    }
    
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.isDirectory() && entry.name !== 'target' && entry.name !== 'node_modules') {
        const subFiles = await this.findRustFiles(fullPath);
        files.push(...subFiles);
      } else if (entry.isFile() && entry.name.endsWith('.rs')) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  /**
   * Check if a file contains test functions
   */
  private async hasTestFunctions(filePath: string): Promise<boolean> {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      return content.includes('#[test]') || content.includes('#[cfg(test)]') || 
             content.includes('#[tokio::test]') || content.includes('#[async_test]');
    } catch {
      return false;
    }
  }

  /**
   * Analyze a test file and extract test functions
   */
  private async analyzeTestFile(filePath: string): Promise<TestModule | null> {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');
      const testFunctions = this.extractTestFunctions(lines);
      
      if (testFunctions.length === 0) {
        return null;
      }

      const relativePath = path.relative(this.projectRoot, filePath);
      const stats = fs.statSync(filePath);
      
      return {
        filePath,
        relativePath,
        moduleType: this.determineModuleType(relativePath),
        testFunctions,
        totalTests: testFunctions.length,
        category: this.determineCategory(relativePath),
        lastModified: stats.mtime
      };
    } catch (error) {
      console.error(`Error analyzing test file ${filePath}:`, error);
      return null;
    }
  }

  /**
   * Extract test functions from file content
   */
  private extractTestFunctions(lines: string[]): TestFunction[] {
    const testFunctions: TestFunction[] = [];
    let currentAttributes: string[] = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Collect attributes
      if (line.startsWith('#[')) {
        currentAttributes.push(line);
        continue;
      }
      
      // Check for test function
      if (line.startsWith('fn ') && this.hasTestAttribute(currentAttributes)) {
        const functionName = this.extractFunctionName(line);
        if (functionName) {
          const isIgnored = currentAttributes.some(attr => attr.includes('ignore'));
          const description = this.extractTestDescription(lines, i);
          
          testFunctions.push({
            name: functionName,
            line: i + 1,
            module: this.extractModuleName(lines),
            isIgnored,
            attributes: [...currentAttributes],
            description
          });
        }
      }
      
      // Reset attributes if we hit a non-attribute, non-function line
      if (!line.startsWith('#[') && !line.startsWith('fn ') && line.length > 0) {
        currentAttributes = [];
      }
    }
    
    return testFunctions;
  }

  /**
   * Check if attributes contain test markers
   */
  private hasTestAttribute(attributes: string[]): boolean {
    return attributes.some(attr => 
      attr.includes('#[test]') || 
      attr.includes('#[tokio::test]') || 
      attr.includes('#[async_test]') ||
      attr.includes('#[bench]')
    );
  }

  /**
   * Extract function name from function declaration
   */
  private extractFunctionName(line: string): string | null {
    const match = line.match(/fn\s+(\w+)/);
    return match ? match[1] : null;
  }

  /**
   * Extract module name from file content
   */
  private extractModuleName(lines: string[]): string {
    for (const line of lines) {
      const match = line.match(/mod\s+(\w+)/);
      if (match) {
        return match[1];
      }
    }
    return 'main';
  }

  /**
   * Extract test description from comments
   */
  private extractTestDescription(lines: string[], functionLineIndex: number): string | undefined {
    // Look for comments above the test function
    for (let i = functionLineIndex - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('//')) {
        return line.replace(/^\/\/\s*/, '');
      }
      if (line.length > 0 && !line.startsWith('#[')) {
        break;
      }
    }
    return undefined;
  }

  /**
   * Determine module type based on file path
   */
  private determineModuleType(relativePath: string): 'unit' | 'integration' | 'benchmark' | 'example' {
    if (relativePath.startsWith('tests/')) {
      if (relativePath.includes('benches/')) return 'benchmark';
      return 'integration';
    }
    if (relativePath.startsWith('examples/')) {
      return 'example';
    }
    return 'unit';
  }

  /**
   * Determine test category based on file path
   */
  private determineCategory(relativePath: string): string {
    if (relativePath.includes('cognitive')) return 'Cognitive Processing';
    if (relativePath.includes('core')) return 'Core Graph Operations';
    if (relativePath.includes('learning')) return 'Learning Algorithms';
    if (relativePath.includes('memory')) return 'Memory Systems';
    if (relativePath.includes('monitoring')) return 'Monitoring & Telemetry';
    if (relativePath.includes('storage')) return 'Storage Systems';
    if (relativePath.includes('query')) return 'Query Processing';
    if (relativePath.includes('embedding')) return 'Embedding Systems';
    if (relativePath.includes('federation')) return 'Federation';
    if (relativePath.includes('gpu')) return 'GPU Operations';
    if (relativePath.includes('mcp')) return 'MCP Integration';
    if (relativePath.includes('extraction')) return 'Data Extraction';
    if (relativePath.startsWith('tests/integration')) return 'Integration Tests';
    if (relativePath.startsWith('tests/stress')) return 'Stress Tests';
    return 'Miscellaneous';
  }

  /**
   * Categorize modules by their categories
   */
  private categorizeModules(modules: TestModule[]): { [key: string]: TestModule[] } {
    const categories: { [key: string]: TestModule[] } = {};
    
    for (const module of modules) {
      if (!categories[module.category]) {
        categories[module.category] = [];
      }
      categories[module.category].push(module);
    }
    
    return categories;
  }

  /**
   * Execute cargo test for specific test patterns
   */
  async executeTests(
    testPattern?: string,
    options: {
      release?: boolean;
      nocapture?: boolean;
      ignored?: boolean;
      features?: string[];
    } = {}
  ): Promise<TestExecutionSummary> {
    const args = ['test'];
    
    if (testPattern) {
      args.push(testPattern);
    }
    
    if (options.release) {
      args.push('--release');
    }
    
    if (options.features && options.features.length > 0) {
      args.push('--features', options.features.join(','));
    }
    
    if (options.nocapture) {
      args.push('--nocapture');
    }
    
    if (options.ignored) {
      args.push('--ignored');
    }
    
    args.push('--', '--format', 'json');
    
    const startTime = Date.now();
    
    try {
      const { stdout, stderr } = await execAsync(`cargo ${args.join(' ')}`, {
        cwd: this.projectRoot,
        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
      });
      
      const executionTime = Date.now() - startTime;
      return this.parseCargoTestOutput(stdout, stderr, executionTime);
    } catch (error: any) {
      const executionTime = Date.now() - startTime;
      // Cargo test returns non-zero exit code when tests fail, but that's expected
      if (error.stdout) {
        return this.parseCargoTestOutput(error.stdout, error.stderr || '', executionTime);
      }
      
      throw new Error(`Test execution failed: ${error.message}`);
    }
  }

  /**
   * Parse cargo test JSON output
   */
  private parseCargoTestOutput(stdout: string, stderr: string, executionTime: number): TestExecutionSummary {
    const results: CargoTestResult[] = [];
    let passed = 0;
    let failed = 0;
    let ignored = 0;
    
    const lines = stdout.split('\n').filter(line => line.trim());
    
    for (const line of lines) {
      try {
        const data = JSON.parse(line);
        
        if (data.type === 'test') {
          const result: CargoTestResult = {
            testName: data.name,
            outcome: data.event as 'passed' | 'failed' | 'ignored' | 'timeout',
            executionTime: data.exec_time || 0,
            stdout: data.stdout,
            stderr: data.stderr
          };
          
          if (data.event === 'failed' && data.stdout) {
            result.failureMessage = this.extractFailureMessage(data.stdout);
          }
          
          results.push(result);
          
          switch (data.event) {
            case 'ok':
              passed++;
              break;
            case 'failed':
              failed++;
              break;
            case 'ignored':
              ignored++;
              break;
          }
        }
      } catch {
        // Skip non-JSON lines
      }
    }
    
    return {
      passed,
      failed,
      ignored,
      total: passed + failed + ignored,
      executionTime,
      results
    };
  }

  /**
   * Extract failure message from test output
   */
  private extractFailureMessage(output: string): string {
    const lines = output.split('\n');
    const failureLines = [];
    let inFailureSection = false;
    
    for (const line of lines) {
      if (line.includes('assertion failed') || line.includes('panicked at')) {
        inFailureSection = true;
      }
      
      if (inFailureSection) {
        failureLines.push(line);
        if (line.trim() === '' && failureLines.length > 1) {
          break;
        }
      }
    }
    
    return failureLines.join('\n').trim();
  }

  /**
   * Get test coverage information (if available)
   */
  async getCoverage(): Promise<any> {
    try {
      // This would require cargo-tarpaulin or similar coverage tool
      const { stdout } = await execAsync('cargo tarpaulin --out Json', {
        cwd: this.projectRoot
      });
      
      return JSON.parse(stdout);
    } catch {
      // Coverage not available
      return null;
    }
  }

  /**
   * Watch for test file changes
   */
  watchTestFiles(callback: (changedFile: string) => void): () => void {
    const chokidar = require('chokidar');
    
    const watcher = chokidar.watch([
      path.join(this.projectRoot, 'src/**/*.rs'),
      path.join(this.projectRoot, 'tests/**/*.rs')
    ], {
      ignored: /target/,
      persistent: true
    });
    
    watcher.on('change', callback);
    
    return () => watcher.close();
  }
}

export const testDiscoveryService = new TestDiscoveryService();