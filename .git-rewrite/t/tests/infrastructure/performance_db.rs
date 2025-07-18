//! Performance Database
//! 
//! SQLite-based storage for performance baselines, historical data, and trend analysis.

use anyhow::{Result, anyhow};
use rusqlite::{Connection, params, Row, Transaction, ToSql};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Performance database for storing baselines and historical data
pub struct PerformanceDatabase {
    /// SQLite connection (thread-safe)
    connection: Arc<Mutex<Connection>>,
    /// Database file path
    db_path: PathBuf,
    /// Configuration for the database
    config: DatabaseConfig,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Maximum number of historical records per test
    pub max_history_per_test: u32,
    /// Data retention period
    pub retention_days: u32,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
    /// Backup configuration
    pub backup_config: Option<BackupConfig>,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Backup interval
    pub backup_interval: Duration,
    /// Maximum number of backups to keep
    pub max_backups: u32,
}

/// Baseline performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetric {
    /// Unique identifier for the baseline
    pub id: i64,
    /// Test name
    pub test_name: String,
    /// Metric name (e.g., "latency_ms", "memory_mb", "cpu_percent")
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f64,
    /// Standard deviation for the baseline
    pub baseline_std_dev: f64,
    /// Tolerance percentage for regression detection
    pub tolerance_percent: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence_level: f64,
    /// Number of samples used to establish baseline
    pub sample_count: u32,
    /// When the baseline was established
    pub established_at: SystemTime,
    /// When the baseline was last updated
    pub updated_at: SystemTime,
    /// Baseline establishment method
    pub establishment_method: EstablishmentMethod,
    /// Historical trend data
    pub trend_history: Vec<TrendPoint>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Method used to establish the baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EstablishmentMethod {
    /// Statistical analysis of historical data
    Statistical {
        /// Statistical method used
        method: StatisticalMethod,
        /// Minimum samples required
        min_samples: u32,
    },
    /// Manual configuration
    Manual {
        /// Who set the baseline
        set_by: String,
        /// Reason for manual setting
        reason: String,
    },
    /// Derived from target specifications
    Target {
        /// Target source
        source: String,
        /// Target type
        target_type: String,
    },
}

/// Statistical methods for baseline establishment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMethod {
    /// Simple mean with standard deviation
    Mean,
    /// Median with interquartile range
    Median,
    /// Trimmed mean (excluding outliers)
    TrimmedMean { trim_percent: f64 },
    /// Percentile-based
    Percentile { percentile: f64 },
}

/// Historical performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    /// Unique record ID
    pub id: i64,
    /// Test name
    pub test_name: String,
    /// Test execution timestamp
    pub timestamp: SystemTime,
    /// Performance metrics for this run
    pub metrics: HashMap<String, f64>,
    /// Test configuration hash
    pub config_hash: String,
    /// Test environment information
    pub environment: TestEnvironment,
    /// Test execution duration
    pub execution_duration: Duration,
    /// Test result status
    pub test_status: TestStatus,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Build information
    pub build_info: BuildInfo,
}

/// Test environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu_info: String,
    /// Total system memory
    pub memory_total: u64,
    /// Rust version
    pub rust_version: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
    /// Target architecture
    pub target_arch: String,
}

/// Test execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Timeout,
    Skipped,
    Error,
}

/// Build information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Build timestamp
    pub build_time: SystemTime,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Feature flags enabled
    pub features: Vec<String>,
    /// Optimization level
    pub opt_level: String,
}

/// Trend analysis point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metric value
    pub value: f64,
    /// Moving average
    pub moving_average: f64,
    /// Trend direction
    pub trend_direction: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Performance comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Test name
    pub test_name: String,
    /// Metric comparisons
    pub metric_comparisons: Vec<MetricComparison>,
    /// Overall regression detected
    pub regression_detected: bool,
    /// Comparison timestamp
    pub timestamp: SystemTime,
}

/// Individual metric comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name
    pub metric_name: String,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Percentage change
    pub change_percent: f64,
    /// Whether change is significant
    pub significant: bool,
    /// Whether change indicates regression
    pub regression: bool,
    /// Confidence level of the comparison
    pub confidence: f64,
}

impl PerformanceDatabase {
    /// Create or open a performance database
    pub async fn new(db_path: &Path) -> Result<Self> {
        let config = DatabaseConfig::default();
        Self::new_with_config(db_path, config).await
    }

    /// Create or open a performance database with custom configuration
    pub async fn new_with_config(db_path: &Path, config: DatabaseConfig) -> Result<Self> {
        // Ensure the directory exists
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| anyhow!("Failed to create database directory: {}", e))?;
        }

        let connection = Connection::open(db_path)
            .map_err(|e| anyhow!("Failed to open database: {}", e))?;

        let db = Self {
            connection: Arc::new(Mutex::new(connection)),
            db_path: db_path.to_path_buf(),
            config,
        };

        db.initialize_schema().await?;
        
        if db.config.auto_cleanup {
            db.cleanup_old_data().await?;
        }

        Ok(db)
    }

    /// Initialize the database schema
    async fn initialize_schema(&self) -> Result<()> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        // Baseline metrics table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS baseline_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                baseline_std_dev REAL NOT NULL DEFAULT 0.0,
                tolerance_percent REAL NOT NULL DEFAULT 5.0,
                confidence_level REAL NOT NULL DEFAULT 0.95,
                sample_count INTEGER NOT NULL DEFAULT 1,
                established_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                establishment_method TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                UNIQUE(test_name, metric_name)
            )",
            [],
        )?;

        // Performance records table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS performance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                metrics TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                environment TEXT NOT NULL,
                execution_duration INTEGER NOT NULL,
                test_status TEXT NOT NULL,
                git_commit TEXT,
                build_info TEXT NOT NULL
            )",
            [],
        )?;

        // Trend history table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trend_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                value REAL NOT NULL,
                moving_average REAL NOT NULL,
                trend_direction TEXT NOT NULL,
                FOREIGN KEY(baseline_id) REFERENCES baseline_metrics(id)
            )",
            [],
        )?;

        // Create indexes for better performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_baseline_test_metric 
             ON baseline_metrics(test_name, metric_name)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_test_timestamp 
             ON performance_records(test_name, timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trend_baseline_timestamp 
             ON trend_history(baseline_id, timestamp)",
            [],
        )?;

        Ok(())
    }

    /// Store a performance record
    pub async fn store_performance_record(&self, record: PerformanceRecord) -> Result<i64> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let metrics_json = serde_json::to_string(&record.metrics)
            .map_err(|e| anyhow!("Failed to serialize metrics: {}", e))?;
        
        let environment_json = serde_json::to_string(&record.environment)
            .map_err(|e| anyhow!("Failed to serialize environment: {}", e))?;
        
        let build_info_json = serde_json::to_string(&record.build_info)
            .map_err(|e| anyhow!("Failed to serialize build info: {}", e))?;

        let timestamp = record.timestamp.duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let execution_duration = record.execution_duration.as_millis() as i64;
        let test_status = serde_json::to_string(&record.test_status)?;

        let id = conn.execute(
            "INSERT INTO performance_records 
             (test_name, timestamp, metrics, config_hash, environment, 
              execution_duration, test_status, git_commit, build_info)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.test_name,
                timestamp,
                metrics_json,
                record.config_hash,
                environment_json,
                execution_duration,
                test_status,
                record.git_commit,
                build_info_json,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get performance records for a test
    pub async fn get_performance_records(&self, test_name: &str, limit: Option<u32>) -> Result<Vec<PerformanceRecord>> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
        let query = format!(
            "SELECT id, test_name, timestamp, metrics, config_hash, environment, 
                    execution_duration, test_status, git_commit, build_info
             FROM performance_records
             WHERE test_name = ?1
             ORDER BY timestamp DESC{}",
            limit_clause
        );

        let mut stmt = conn.prepare(&query)?;
        let record_iter = stmt.query_map([test_name], |row| {
            self.row_to_performance_record(row)
        })?;

        let mut records = Vec::new();
        for record in record_iter {
            records.push(record?);
        }

        Ok(records)
    }

    /// Establish a baseline from historical data
    pub async fn establish_baseline(
        &self,
        test_name: &str,
        metric_name: &str,
        method: EstablishmentMethod,
        tolerance_percent: f64,
    ) -> Result<BaselineMetric> {
        // Get recent performance records
        let records = self.get_performance_records(test_name, Some(100)).await?;
        
        if records.is_empty() {
            return Err(anyhow!("No performance records found for test {}", test_name));
        }

        // Extract metric values
        let mut values: Vec<f64> = records.iter()
            .filter_map(|r| r.metrics.get(metric_name).copied())
            .collect();

        if values.is_empty() {
            return Err(anyhow!("No values found for metric {} in test {}", metric_name, test_name));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate baseline value and standard deviation based on method
        let (baseline_value, std_dev) = match &method {
            EstablishmentMethod::Statistical { method: StatisticalMethod::Mean, .. } => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                (mean, variance.sqrt())
            }
            EstablishmentMethod::Statistical { method: StatisticalMethod::Median, .. } => {
                let median = values[values.len() / 2];
                let q1 = values[values.len() / 4];
                let q3 = values[3 * values.len() / 4];
                let iqr = q3 - q1;
                (median, iqr / 1.35) // Approximate std dev from IQR
            }
            EstablishmentMethod::Statistical { method: StatisticalMethod::TrimmedMean { trim_percent }, .. } => {
                let trim_count = (values.len() as f64 * trim_percent / 100.0) as usize;
                let trimmed = &values[trim_count..values.len() - trim_count];
                let mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
                let variance = trimmed.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / trimmed.len() as f64;
                (mean, variance.sqrt())
            }
            EstablishmentMethod::Statistical { method: StatisticalMethod::Percentile { percentile }, .. } => {
                let index = (values.len() as f64 * percentile / 100.0) as usize;
                let value = values[index.min(values.len() - 1)];
                (value, 0.0) // No std dev for percentile
            }
            _ => {
                // For manual and target methods, use simple mean
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                (mean, variance.sqrt())
            }
        };

        let baseline = BaselineMetric {
            id: 0, // Will be set by database
            test_name: test_name.to_string(),
            metric_name: metric_name.to_string(),
            baseline_value,
            baseline_std_dev: std_dev,
            tolerance_percent,
            confidence_level: 0.95,
            sample_count: values.len() as u32,
            established_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            establishment_method: method,
            trend_history: Vec::new(),
            metadata: HashMap::new(),
        };

        self.store_baseline(&baseline).await
    }

    /// Store a baseline metric
    pub async fn store_baseline(&self, baseline: &BaselineMetric) -> Result<BaselineMetric> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let established_at = baseline.established_at.duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let updated_at = baseline.updated_at.duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let establishment_method = serde_json::to_string(&baseline.establishment_method)?;
        let metadata = serde_json::to_string(&baseline.metadata)?;

        let id = conn.execute(
            "INSERT OR REPLACE INTO baseline_metrics 
             (test_name, metric_name, baseline_value, baseline_std_dev, tolerance_percent,
              confidence_level, sample_count, established_at, updated_at, 
              establishment_method, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                baseline.test_name,
                baseline.metric_name,
                baseline.baseline_value,
                baseline.baseline_std_dev,
                baseline.tolerance_percent,
                baseline.confidence_level,
                baseline.sample_count,
                established_at,
                updated_at,
                establishment_method,
                metadata,
            ],
        )?;

        let mut updated_baseline = baseline.clone();
        updated_baseline.id = conn.last_insert_rowid();
        
        Ok(updated_baseline)
    }

    /// Get baseline for a specific test and metric
    pub async fn get_baseline(&self, test_name: &str, metric_name: &str) -> Result<Option<BaselineMetric>> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let mut stmt = conn.prepare(
            "SELECT id, test_name, metric_name, baseline_value, baseline_std_dev,
                    tolerance_percent, confidence_level, sample_count, established_at,
                    updated_at, establishment_method, metadata
             FROM baseline_metrics
             WHERE test_name = ?1 AND metric_name = ?2"
        )?;

        let baseline = stmt.query_row([test_name, metric_name], |row| {
            self.row_to_baseline_metric(row)
        }).optional()?;

        Ok(baseline)
    }

    /// Get all baselines
    pub async fn get_all_baselines(&self) -> Result<Vec<BaselineMetric>> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let mut stmt = conn.prepare(
            "SELECT id, test_name, metric_name, baseline_value, baseline_std_dev,
                    tolerance_percent, confidence_level, sample_count, established_at,
                    updated_at, establishment_method, metadata
             FROM baseline_metrics
             ORDER BY test_name, metric_name"
        )?;

        let baseline_iter = stmt.query_map([], |row| {
            self.row_to_baseline_metric(row)
        })?;

        let mut baselines = Vec::new();
        for baseline in baseline_iter {
            baselines.push(baseline?);
        }

        Ok(baselines)
    }

    /// Compare current metrics against baselines
    pub async fn compare_against_baseline(
        &self,
        test_name: &str,
        current_metrics: &HashMap<String, f64>,
    ) -> Result<PerformanceComparison> {
        let mut metric_comparisons = Vec::new();
        let mut regression_detected = false;

        for (metric_name, &current_value) in current_metrics {
            if let Some(baseline) = self.get_baseline(test_name, metric_name).await? {
                let change_percent = if baseline.baseline_value != 0.0 {
                    ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100.0
                } else {
                    0.0
                };

                let significant = change_percent.abs() > baseline.tolerance_percent;
                let regression = significant && change_percent > 0.0; // Assuming higher is worse

                if regression {
                    regression_detected = true;
                }

                let comparison = MetricComparison {
                    metric_name: metric_name.clone(),
                    baseline_value: baseline.baseline_value,
                    current_value,
                    change_percent,
                    significant,
                    regression,
                    confidence: baseline.confidence_level,
                };

                metric_comparisons.push(comparison);
            }
        }

        Ok(PerformanceComparison {
            test_name: test_name.to_string(),
            metric_comparisons,
            regression_detected,
            timestamp: SystemTime::now(),
        })
    }

    /// Update baselines from test results
    pub async fn update_baselines(&self, _results: &crate::infrastructure::TestReport) -> Result<()> {
        // This would analyze test results and update baselines accordingly
        // Implementation would depend on TestReport structure
        Ok(())
    }

    /// Cleanup old data based on retention policy
    pub async fn cleanup_old_data(&self) -> Result<u32> {
        let conn = self.connection.lock()
            .map_err(|_| anyhow!("Failed to acquire database connection"))?;

        let cutoff_time = SystemTime::now() - Duration::from_secs(self.config.retention_days as u64 * 24 * 3600);
        let cutoff_timestamp = cutoff_time.duration_since(UNIX_EPOCH)?.as_secs() as i64;

        let deleted_records = conn.execute(
            "DELETE FROM performance_records WHERE timestamp < ?1",
            [cutoff_timestamp],
        )?;

        // Also cleanup trend history for non-existent baselines
        conn.execute(
            "DELETE FROM trend_history 
             WHERE baseline_id NOT IN (SELECT id FROM baseline_metrics)",
            [],
        )?;

        Ok(deleted_records as u32)
    }

    /// Export database to JSON
    pub async fn export_to_json(&self) -> Result<String> {
        let baselines = self.get_all_baselines().await?;
        
        #[derive(Serialize)]
        struct DatabaseExport {
            baselines: Vec<BaselineMetric>,
            export_timestamp: SystemTime,
            database_path: PathBuf,
        }

        let export = DatabaseExport {
            baselines,
            export_timestamp: SystemTime::now(),
            database_path: self.db_path.clone(),
        };

        serde_json::to_string_pretty(&export)
            .map_err(|e| anyhow!("Failed to serialize database export: {}", e))
    }

    /// Close database connection
    pub async fn close(&self) -> Result<()> {
        // Connection will be dropped and closed automatically
        Ok(())
    }

    /// Helper function to convert database row to BaselineMetric
    fn row_to_baseline_metric(&self, row: &Row) -> rusqlite::Result<BaselineMetric> {
        let established_at_secs: i64 = row.get(8)?;
        let updated_at_secs: i64 = row.get(9)?;
        let establishment_method_json: String = row.get(10)?;
        let metadata_json: String = row.get(11)?;

        let established_at = UNIX_EPOCH + Duration::from_secs(established_at_secs as u64);
        let updated_at = UNIX_EPOCH + Duration::from_secs(updated_at_secs as u64);
        
        let establishment_method: EstablishmentMethod = serde_json::from_str(&establishment_method_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(10, rusqlite::types::Type::Text, Box::new(e)))?;
        
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(11, rusqlite::types::Type::Text, Box::new(e)))?;

        Ok(BaselineMetric {
            id: row.get(0)?,
            test_name: row.get(1)?,
            metric_name: row.get(2)?,
            baseline_value: row.get(3)?,
            baseline_std_dev: row.get(4)?,
            tolerance_percent: row.get(5)?,
            confidence_level: row.get(6)?,
            sample_count: row.get(7)?,
            established_at,
            updated_at,
            establishment_method,
            trend_history: Vec::new(), // Would be loaded separately if needed
            metadata,
        })
    }

    /// Helper function to convert database row to PerformanceRecord
    fn row_to_performance_record(&self, row: &Row) -> rusqlite::Result<PerformanceRecord> {
        let timestamp_secs: i64 = row.get(2)?;
        let metrics_json: String = row.get(3)?;
        let environment_json: String = row.get(5)?;
        let execution_duration_ms: i64 = row.get(6)?;
        let test_status_json: String = row.get(7)?;
        let build_info_json: String = row.get(9)?;

        let timestamp = UNIX_EPOCH + Duration::from_secs(timestamp_secs as u64);
        let execution_duration = Duration::from_millis(execution_duration_ms as u64);
        
        let metrics: HashMap<String, f64> = serde_json::from_str(&metrics_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e)))?;
        
        let environment: TestEnvironment = serde_json::from_str(&environment_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(e)))?;
        
        let test_status: TestStatus = serde_json::from_str(&test_status_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(e)))?;
        
        let build_info: BuildInfo = serde_json::from_str(&build_info_json)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(9, rusqlite::types::Type::Text, Box::new(e)))?;

        Ok(PerformanceRecord {
            id: row.get(0)?,
            test_name: row.get(1)?,
            timestamp,
            metrics,
            config_hash: row.get(4)?,
            environment,
            execution_duration,
            test_status,
            git_commit: row.get(8)?,
            build_info,
        })
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            max_history_per_test: 1000,
            retention_days: 90,
            auto_cleanup: true,
            backup_config: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_database_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let db = PerformanceDatabase::new(&db_path).await.unwrap();
        assert!(db_path.exists());
    }

    #[tokio::test]
    async fn test_baseline_operations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = PerformanceDatabase::new(&db_path).await.unwrap();

        let baseline = BaselineMetric {
            id: 0,
            test_name: "test_example".to_string(),
            metric_name: "latency_ms".to_string(),
            baseline_value: 5.0,
            baseline_std_dev: 1.0,
            tolerance_percent: 10.0,
            confidence_level: 0.95,
            sample_count: 100,
            established_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            establishment_method: EstablishmentMethod::Statistical {
                method: StatisticalMethod::Mean,
                min_samples: 10,
            },
            trend_history: Vec::new(),
            metadata: HashMap::new(),
        };

        // Store baseline
        let stored = db.store_baseline(&baseline).await.unwrap();
        assert!(stored.id > 0);

        // Retrieve baseline
        let retrieved = db.get_baseline("test_example", "latency_ms").await.unwrap().unwrap();
        assert_eq!(retrieved.baseline_value, 5.0);
        assert_eq!(retrieved.test_name, "test_example");

        // Get all baselines
        let all_baselines = db.get_all_baselines().await.unwrap();
        assert_eq!(all_baselines.len(), 1);
    }

    #[tokio::test]
    async fn test_performance_records() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = PerformanceDatabase::new(&db_path).await.unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("latency_ms".to_string(), 5.5);
        metrics.insert("memory_mb".to_string(), 100.0);

        let record = PerformanceRecord {
            id: 0,
            test_name: "test_example".to_string(),
            timestamp: SystemTime::now(),
            metrics,
            config_hash: "abc123".to_string(),
            environment: TestEnvironment {
                os: "Linux".to_string(),
                cpu_info: "Intel Core i7".to_string(),
                memory_total: 16 * 1024 * 1024 * 1024,
                rust_version: "1.70.0".to_string(),
                compiler_flags: vec!["--release".to_string()],
                target_arch: "x86_64".to_string(),
            },
            execution_duration: Duration::from_millis(1000),
            test_status: TestStatus::Passed,
            git_commit: Some("def456".to_string()),
            build_info: BuildInfo {
                build_time: SystemTime::now(),
                build_config: "release".to_string(),
                features: vec!["default".to_string()],
                opt_level: "s".to_string(),
            },
        };

        // Store record
        let record_id = db.store_performance_record(record).await.unwrap();
        assert!(record_id > 0);

        // Retrieve records
        let records = db.get_performance_records("test_example", Some(10)).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].test_name, "test_example");
    }

    #[tokio::test]
    async fn test_baseline_comparison() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = PerformanceDatabase::new(&db_path).await.unwrap();

        // Store a baseline
        let baseline = BaselineMetric {
            id: 0,
            test_name: "test_example".to_string(),
            metric_name: "latency_ms".to_string(),
            baseline_value: 5.0,
            baseline_std_dev: 1.0,
            tolerance_percent: 10.0,
            confidence_level: 0.95,
            sample_count: 100,
            established_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            establishment_method: EstablishmentMethod::Statistical {
                method: StatisticalMethod::Mean,
                min_samples: 10,
            },
            trend_history: Vec::new(),
            metadata: HashMap::new(),
        };

        db.store_baseline(&baseline).await.unwrap();

        // Compare against baseline
        let mut current_metrics = HashMap::new();
        current_metrics.insert("latency_ms".to_string(), 6.0); // 20% increase

        let comparison = db.compare_against_baseline("test_example", &current_metrics).await.unwrap();
        
        assert_eq!(comparison.metric_comparisons.len(), 1);
        assert!(comparison.metric_comparisons[0].significant);
        assert!(comparison.metric_comparisons[0].regression);
        assert!(comparison.regression_detected);
    }

    #[tokio::test]
    async fn test_data_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let mut config = DatabaseConfig::default();
        config.retention_days = 0; // Immediate cleanup
        
        let db = PerformanceDatabase::new_with_config(&db_path, config).await.unwrap();

        // Store a record that should be cleaned up
        let mut metrics = HashMap::new();
        metrics.insert("latency_ms".to_string(), 5.0);

        let record = PerformanceRecord {
            id: 0,
            test_name: "test_example".to_string(),
            timestamp: SystemTime::now() - Duration::from_secs(3600), // 1 hour ago
            metrics,
            config_hash: "abc123".to_string(),
            environment: TestEnvironment {
                os: "Linux".to_string(),
                cpu_info: "Intel Core i7".to_string(),
                memory_total: 16 * 1024 * 1024 * 1024,
                rust_version: "1.70.0".to_string(),
                compiler_flags: vec!["--release".to_string()],
                target_arch: "x86_64".to_string(),
            },
            execution_duration: Duration::from_millis(1000),
            test_status: TestStatus::Passed,
            git_commit: None,
            build_info: BuildInfo {
                build_time: SystemTime::now(),
                build_config: "release".to_string(),
                features: vec!["default".to_string()],
                opt_level: "s".to_string(),
            },
        };

        db.store_performance_record(record).await.unwrap();

        // Cleanup should remove the old record
        let deleted_count = db.cleanup_old_data().await.unwrap();
        assert_eq!(deleted_count, 1);

        let records = db.get_performance_records("test_example", None).await.unwrap();
        assert_eq!(records.len(), 0);
    }

    #[tokio::test]
    async fn test_json_export() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let db = PerformanceDatabase::new(&db_path).await.unwrap();

        let baseline = BaselineMetric {
            id: 0,
            test_name: "test_example".to_string(),
            metric_name: "latency_ms".to_string(),
            baseline_value: 5.0,
            baseline_std_dev: 1.0,
            tolerance_percent: 10.0,
            confidence_level: 0.95,
            sample_count: 100,
            established_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            establishment_method: EstablishmentMethod::Statistical {
                method: StatisticalMethod::Mean,
                min_samples: 10,
            },
            trend_history: Vec::new(),
            metadata: HashMap::new(),
        };

        db.store_baseline(&baseline).await.unwrap();

        let json_export = db.export_to_json().await.unwrap();
        assert!(json_export.contains("test_example"));
        assert!(json_export.contains("latency_ms"));
        assert!(json_export.contains("baselines"));
    }
}