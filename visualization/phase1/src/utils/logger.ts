/**
 * Logging utility for LLMKG Visualization System
 * 
 * Provides structured logging with different levels and contexts
 * for debugging and monitoring the WebSocket communication system.
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

export interface LogEntry {
  timestamp: number;
  level: LogLevel;
  context: string;
  message: string;
  data?: any;
}

export class Logger {
  private context: string;
  private static globalLevel: LogLevel = LogLevel.INFO;
  private static logs: LogEntry[] = [];
  private static maxLogs: number = 1000;

  constructor(context: string) {
    this.context = context;
  }

  /**
   * Set global log level
   */
  static setLevel(level: LogLevel): void {
    Logger.globalLevel = level;
  }

  /**
   * Get all log entries
   */
  static getLogs(): LogEntry[] {
    return [...Logger.logs];
  }

  /**
   * Clear all logs
   */
  static clearLogs(): void {
    Logger.logs = [];
  }

  /**
   * Debug level logging
   */
  debug(message: string, data?: any): void {
    this.log(LogLevel.DEBUG, message, data);
  }

  /**
   * Info level logging
   */
  info(message: string, data?: any): void {
    this.log(LogLevel.INFO, message, data);
  }

  /**
   * Warning level logging
   */
  warn(message: string, data?: any): void {
    this.log(LogLevel.WARN, message, data);
  }

  /**
   * Error level logging
   */
  error(message: string, error?: any): void {
    this.log(LogLevel.ERROR, message, error);
  }

  /**
   * Core logging method
   */
  private log(level: LogLevel, message: string, data?: any): void {
    if (level < Logger.globalLevel) {
      return;
    }

    const logEntry: LogEntry = {
      timestamp: Date.now(),
      level,
      context: this.context,
      message,
      data
    };

    // Add to logs array
    Logger.logs.push(logEntry);

    // Trim logs if needed
    if (Logger.logs.length > Logger.maxLogs) {
      Logger.logs.shift();
    }

    // Console output
    const timestamp = new Date(logEntry.timestamp).toISOString();
    const levelName = LogLevel[level];
    const prefix = `[${timestamp}] [${levelName}] [${this.context}]`;

    switch (level) {
      case LogLevel.DEBUG:
        if (data) {
          console.debug(prefix, message, data);
        } else {
          console.debug(prefix, message);
        }
        break;
      case LogLevel.INFO:
        if (data) {
          console.info(prefix, message, data);
        } else {
          console.info(prefix, message);
        }
        break;
      case LogLevel.WARN:
        if (data) {
          console.warn(prefix, message, data);
        } else {
          console.warn(prefix, message);
        }
        break;
      case LogLevel.ERROR:
        if (data) {
          console.error(prefix, message, data);
        } else {
          console.error(prefix, message);
        }
        break;
    }
  }
}

// Export a default logger instance
export const defaultLogger = new Logger('LLMKG');