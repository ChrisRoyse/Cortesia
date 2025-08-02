/*!
Phase 5.4: Metrics Exporters
Export metrics to various monitoring systems (Prometheus, InfluxDB, JSON, etc.)
*/

use crate::monitoring::metrics::{MetricRegistry, MetricSample, MetricValue};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub enabled: bool,
    pub export_interval: Duration,
    pub batch_size: usize,
    pub timeout: Duration,
    pub retry_attempts: usize,
    pub retry_delay: Duration,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_interval: Duration::from_secs(30),
            batch_size: 1000,
            timeout: Duration::from_secs(10),
            retry_attempts: 3,
            retry_delay: Duration::from_secs(1),
        }
    }
}

#[async_trait::async_trait]
pub trait MetricsExporter: Send + Sync {
    async fn export(&self, samples: Vec<MetricSample>) -> Result<(), Box<dyn std::error::Error>>;
    fn name(&self) -> &str;
    fn is_healthy(&self) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    pub push_gateway_url: String,
    pub job_name: String,
    pub instance: String,
    pub basic_auth: Option<BasicAuth>,
    pub extra_labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicAuth {
    pub username: String,
    pub password: String,
}

pub struct PrometheusExporter {
    config: PrometheusConfig,
    export_config: ExportConfig,
    client: reqwest::Client,
}

impl PrometheusExporter {
    pub fn new(config: PrometheusConfig, export_config: ExportConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(export_config.timeout)
            .build()
            .unwrap();
        
        Self {
            config,
            export_config,
            client,
        }
    }
    
    fn format_prometheus_metrics(&self, samples: Vec<MetricSample>) -> String {
        let mut output = String::new();
        
        for sample in samples {
            let metric_name = self.sanitize_metric_name(&sample.name);
            let labels = self.format_labels(&sample.labels);
            
            match sample.value {
                MetricValue::Counter(value) => {
                    output.push_str(&format!(
                        "# TYPE {metric_name} counter\n{metric_name}{labels} {value}\n"
                    ));
                }
                MetricValue::Gauge(value) => {
                    output.push_str(&format!(
                        "# TYPE {metric_name} gauge\n{metric_name}{labels} {value}\n"
                    ));
                }
                MetricValue::Histogram { count, sum, buckets } => {
                    output.push_str(&format!("# TYPE {metric_name} histogram\n"));
                    
                    // Histogram buckets
                    for (upper_bound, bucket_count) in buckets {
                        let bucket_labels = self.format_histogram_bucket_labels(&sample.labels, upper_bound);
                        output.push_str(&format!(
                            "{metric_name}_bucket{bucket_labels} {bucket_count}\n"
                        ));
                    }
                    
                    // Histogram count and sum
                    output.push_str(&format!("{metric_name}_count{labels} {count}\n"));
                    output.push_str(&format!("{metric_name}_sum{labels} {sum}\n"));
                }
                MetricValue::Timer { count, sum_duration_ms, percentiles, .. } => {
                    output.push_str(&format!("# TYPE {metric_name} histogram\n"));
                    
                    // Timer percentiles as gauges
                    for (percentile_name, value) in percentiles {
                        let percentile_labels = self.format_percentile_labels(&sample.labels, &percentile_name);
                        output.push_str(&format!(
                            "{metric_name}_percentile{percentile_labels} {value}\n"
                        ));
                    }
                    
                    output.push_str(&format!("{metric_name}_count{labels} {count}\n"));
                    output.push_str(&format!("{metric_name}_sum{labels} {sum_duration_ms}\n"));
                }
                MetricValue::Summary { count, sum, quantiles } => {
                    output.push_str(&format!("# TYPE {metric_name} summary\n"));
                    
                    for (quantile_name, value) in quantiles {
                        let quantile_labels = self.format_quantile_labels(&sample.labels, &quantile_name);
                        output.push_str(&format!(
                            "{metric_name}{quantile_labels} {value}\n"
                        ));
                    }
                    
                    output.push_str(&format!("{metric_name}_count{labels} {count}\n"));
                    output.push_str(&format!("{metric_name}_sum{labels} {sum}\n"));
                }
            }
        }
        
        output
    }
    
    fn sanitize_metric_name(&self, name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    }
    
    fn format_labels(&self, labels: &HashMap<String, String>) -> String {
        if labels.is_empty() && self.config.extra_labels.is_empty() {
            return String::new();
        }
        
        let mut all_labels = labels.clone();
        all_labels.extend(self.config.extra_labels.clone());
        
        let mut label_pairs: Vec<_> = all_labels.iter().collect();
        label_pairs.sort_by_key(|(k, _)| *k);
        
        let label_string = label_pairs
            .iter()
            .map(|(k, v)| format!("{k}=\"{v}\""))
            .collect::<Vec<_>>()
            .join(",");
        
        format!("{{{label_string}}}")
    }
    
    fn format_histogram_bucket_labels(&self, labels: &HashMap<String, String>, upper_bound: f64) -> String {
        let mut bucket_labels = labels.clone();
        bucket_labels.insert("le".to_string(), 
            if upper_bound.is_infinite() { "+Inf".to_string() } else { upper_bound.to_string() });
        bucket_labels.extend(self.config.extra_labels.clone());
        
        self.format_labels(&bucket_labels)
    }
    
    fn format_percentile_labels(&self, labels: &HashMap<String, String>, percentile: &str) -> String {
        let mut percentile_labels = labels.clone();
        percentile_labels.insert("percentile".to_string(), percentile.to_string());
        percentile_labels.extend(self.config.extra_labels.clone());
        
        self.format_labels(&percentile_labels)
    }
    
    fn format_quantile_labels(&self, labels: &HashMap<String, String>, quantile: &str) -> String {
        let mut quantile_labels = labels.clone();
        quantile_labels.insert("quantile".to_string(), quantile.to_string());
        quantile_labels.extend(self.config.extra_labels.clone());
        
        self.format_labels(&quantile_labels)
    }
}

#[async_trait::async_trait]
impl MetricsExporter for PrometheusExporter {
    async fn export(&self, samples: Vec<MetricSample>) -> Result<(), Box<dyn std::error::Error>> {
        let metrics_data = self.format_prometheus_metrics(samples);
        
        let url = format!("{}/metrics/job/{}/instance/{}", 
            self.config.push_gateway_url, 
            self.config.job_name, 
            self.config.instance
        );
        
        let mut request = self.client.post(&url)
            .header("Content-Type", "text/plain; version=0.0.4")
            .body(metrics_data);
        
        if let Some(ref auth) = self.config.basic_auth {
            request = request.basic_auth(&auth.username, Some(&auth.password));
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(format!("Prometheus export failed with status: {}", response.status()).into());
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "prometheus"
    }
    
    fn is_healthy(&self) -> bool {
        // Simple health check - could be enhanced to ping the push gateway
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluxDBConfig {
    pub url: String,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub retention_policy: Option<String>,
    pub precision: String, // "s", "ms", "us", "ns"
}

impl Default for InfluxDBConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8086".to_string(),
            database: "llmkg_metrics".to_string(),
            username: None,
            password: None,
            retention_policy: None,
            precision: "ms".to_string(),
        }
    }
}

pub struct InfluxDBExporter {
    config: InfluxDBConfig,
    export_config: ExportConfig,
    client: reqwest::Client,
}

impl InfluxDBExporter {
    pub fn new(config: InfluxDBConfig, export_config: ExportConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(export_config.timeout)
            .build()
            .unwrap();
        
        Self {
            config,
            export_config,
            client,
        }
    }
    
    fn format_influxdb_line_protocol(&self, samples: Vec<MetricSample>) -> String {
        let mut lines = Vec::new();
        
        for sample in samples {
            let measurement = self.sanitize_measurement_name(&sample.name);
            let tags = self.format_tags(&sample.labels);
            let fields = self.format_fields(&sample.value);
            let timestamp = sample.timestamp * 1000; // Convert to milliseconds
            
            if !fields.is_empty() {
                let line = if tags.is_empty() {
                    format!("{measurement} {fields} {timestamp}")
                } else {
                    format!("{measurement},{tags} {fields} {timestamp}")
                };
                lines.push(line);
            }
        }
        
        lines.join("\n")
    }
    
    fn sanitize_measurement_name(&self, name: &str) -> String {
        name.replace(" ", "_").replace(",", "_").replace("=", "_")
    }
    
    fn format_tags(&self, labels: &HashMap<String, String>) -> String {
        if labels.is_empty() {
            return String::new();
        }
        
        let mut tag_pairs: Vec<_> = labels.iter().collect();
        tag_pairs.sort_by_key(|(k, _)| *k);
        
        tag_pairs
            .iter()
            .map(|(k, v)| format!("{}={}", self.escape_tag_key(k), self.escape_tag_value(v)))
            .collect::<Vec<_>>()
            .join(",")
    }
    
    fn format_fields(&self, value: &MetricValue) -> String {
        match value {
            MetricValue::Counter(val) => format!("value={val}i"),
            MetricValue::Gauge(val) => format!("value={val}"),
            MetricValue::Histogram { count, sum, .. } => {
                format!("count={count}i,sum={sum}")
            }
            MetricValue::Timer { count, sum_duration_ms, min_ms, max_ms, percentiles } => {
                let mut fields = vec![
                    format!("count={}i", count),
                    format!("sum={}", sum_duration_ms),
                    format!("min={}", min_ms),
                    format!("max={}", max_ms),
                ];
                
                for (percentile, value) in percentiles {
                    fields.push(format!("{percentile}={value}"));
                }
                
                fields.join(",")
            }
            MetricValue::Summary { count, sum, quantiles } => {
                let mut fields = vec![
                    format!("count={}i", count),
                    format!("sum={}", sum),
                ];
                
                for (quantile, value) in quantiles {
                    fields.push(format!("{quantile}={value}"));
                }
                
                fields.join(",")
            }
        }
    }
    
    fn escape_tag_key(&self, key: &str) -> String {
        key.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")
    }
    
    fn escape_tag_value(&self, value: &str) -> String {
        value.replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")
    }
}

#[async_trait::async_trait]
impl MetricsExporter for InfluxDBExporter {
    async fn export(&self, samples: Vec<MetricSample>) -> Result<(), Box<dyn std::error::Error>> {
        let line_protocol = self.format_influxdb_line_protocol(samples);
        
        let mut url = format!("{}/write?db={}&precision={}", 
            self.config.url, 
            self.config.database, 
            self.config.precision
        );
        
        if let Some(ref rp) = self.config.retention_policy {
            url.push_str(&format!("&rp={rp}"));
        }
        
        let mut request = self.client.post(&url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(line_protocol);
        
        if let (Some(ref username), Some(ref password)) = (&self.config.username, &self.config.password) {
            request = request.basic_auth(username, Some(password));
        }
        
        let response = request.send().await?;
        
        if !response.status().is_success() {
            return Err(format!("InfluxDB export failed with status: {}", response.status()).into());
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "influxdb"
    }
    
    fn is_healthy(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonExportConfig {
    pub output_file: String,
    pub pretty_print: bool,
    pub append_mode: bool,
    pub max_file_size_mb: usize,
    pub rotation_count: usize,
}

impl Default for JsonExportConfig {
    fn default() -> Self {
        Self {
            output_file: "metrics.json".to_string(),
            pretty_print: true,
            append_mode: false,
            max_file_size_mb: 100,
            rotation_count: 5,
        }
    }
}

pub struct JsonExporter {
    config: JsonExportConfig,
    export_config: ExportConfig,
}

impl JsonExporter {
    pub fn new(config: JsonExportConfig, export_config: ExportConfig) -> Self {
        Self {
            config,
            export_config,
        }
    }
    
    async fn rotate_file_if_needed(&self) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::path::Path;
        
        let path = Path::new(&self.config.output_file);
        
        if !path.exists() {
            return Ok(());
        }
        
        let metadata = fs::metadata(path)?;
        let file_size_mb = metadata.len() / (1024 * 1024);
        
        if file_size_mb >= self.config.max_file_size_mb as u64 {
            // Rotate files
            for i in (1..self.config.rotation_count).rev() {
                let old_file = format!("{}.{}", self.config.output_file, i);
                let new_file = format!("{}.{}", self.config.output_file, i + 1);
                
                if Path::new(&old_file).exists() {
                    fs::rename(&old_file, &new_file)?;
                }
            }
            
            let first_backup = format!("{}.1", self.config.output_file);
            fs::rename(&self.config.output_file, &first_backup)?;
        }
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl MetricsExporter for JsonExporter {
    async fn export(&self, samples: Vec<MetricSample>) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        if !self.config.append_mode {
            self.rotate_file_if_needed().await?;
        }
        
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(self.config.append_mode)
            .truncate(!self.config.append_mode)
            .open(&self.config.output_file)?;
        
        let export_data = serde_json::json!({
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            "metrics": samples
        });
        
        let json_str = if self.config.pretty_print {
            serde_json::to_string_pretty(&export_data)?
        } else {
            serde_json::to_string(&export_data)?
        };
        
        if self.config.append_mode {
            writeln!(file, "{json_str}")?;
        } else {
            write!(file, "{json_str}")?;
        }
        
        file.flush()?;
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "json"
    }
    
    fn is_healthy(&self) -> bool {
        use std::fs::OpenOptions;
        
        // Test if we can write to the output file
        OpenOptions::new()
            .create(true)
            
            .append(true)
            .open(&self.config.output_file)
            .is_ok()
    }
}

pub struct MultiExporter {
    exporters: Vec<Box<dyn MetricsExporter>>,
    export_config: ExportConfig,
}

impl MultiExporter {
    pub fn new(exporters: Vec<Box<dyn MetricsExporter>>, export_config: ExportConfig) -> Self {
        Self {
            exporters,
            export_config,
        }
    }
    
    pub fn add_exporter(&mut self, exporter: Box<dyn MetricsExporter>) {
        self.exporters.push(exporter);
    }
    
    pub async fn start_background_export(&self, registry: Arc<MetricRegistry>) {
        let exporters: Vec<_> = self.exporters.iter().map(|e| e.name().to_string()).collect();
        println!("Starting background metrics export with exporters: {exporters:?}");
        
        let mut interval = interval(self.export_config.export_interval);
        let batch_size = self.export_config.batch_size;
        
        loop {
            interval.tick().await;
            
            let samples = registry.collect_all_samples();
            
            // Process samples in batches
            for batch in samples.chunks(batch_size) {
                let batch_samples = batch.to_vec();
                
                // Export to all configured exporters in parallel
                let export_tasks: Vec<_> = self.exporters.iter().map(|exporter| {
                    let samples_clone = batch_samples.clone();
                    async move {
                        let exporter_name = exporter.name();
                        match exporter.export(samples_clone).await {
                            Ok(_) => {
                                println!("Successfully exported metrics to {exporter_name}");
                            }
                            Err(e) => {
                                eprintln!("Failed to export metrics to {exporter_name}: {e}");
                            }
                        }
                    }
                }).collect();
                
                // Wait for all exports to complete
                futures::future::join_all(export_tasks).await;
            }
        }
    }
}

#[async_trait::async_trait]
impl MetricsExporter for MultiExporter {
    async fn export(&self, samples: Vec<MetricSample>) -> Result<(), Box<dyn std::error::Error>> {
        let mut errors = Vec::new();
        
        for exporter in &self.exporters {
            if let Err(e) = exporter.export(samples.clone()).await {
                errors.push(format!("{}: {}", exporter.name(), e));
            }
        }
        
        if !errors.is_empty() {
            return Err(format!("Export errors: {}", errors.join(", ")).into());
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "multi"
    }
    
    fn is_healthy(&self) -> bool {
        self.exporters.iter().all(|e| e.is_healthy())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitoring::metrics::MetricValue;
    
    #[test]
    fn test_prometheus_format() {
        let config = PrometheusConfig {
            push_gateway_url: "http://localhost:9091".to_string(),
            job_name: "llmkg".to_string(),
            instance: "test".to_string(),
            basic_auth: None,
            extra_labels: HashMap::new(),
        };
        
        let exporter = PrometheusExporter::new(config, ExportConfig::default());
        
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        
        let sample = MetricSample {
            name: "http_requests_total".to_string(),
            value: MetricValue::Counter(42),
            labels,
            timestamp: 1640995200,
            help: None,
        };
        
        let formatted = exporter.format_prometheus_metrics(vec![sample]);
        
        assert!(formatted.contains("# TYPE http_requests_total counter"));
        assert!(formatted.contains("http_requests_total{method=\"GET\"} 42"));
    }
    
    #[test]
    fn test_influxdb_format() {
        let config = InfluxDBConfig::default();
        let exporter = InfluxDBExporter::new(config, ExportConfig::default());
        
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        
        let sample = MetricSample {
            name: "http_requests_total".to_string(),
            value: MetricValue::Counter(42),
            labels,
            timestamp: 1640995200,
            help: None,
        };
        
        let formatted = exporter.format_influxdb_line_protocol(vec![sample]);
        
        assert!(formatted.contains("http_requests_total,method=GET value=42i 1640995200000"));
    }
    
    #[tokio::test]
    async fn test_json_exporter() {
        use tempfile::NamedTempFile;
        
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_string_lossy().to_string();
        
        let config = JsonExportConfig {
            output_file: file_path.clone(),
            pretty_print: false,
            append_mode: false,
            max_file_size_mb: 1,
            rotation_count: 3,
        };
        
        let exporter = JsonExporter::new(config, ExportConfig::default());
        
        let sample = MetricSample {
            name: "test_metric".to_string(),
            value: MetricValue::Gauge(3.14),
            labels: HashMap::new(),
            timestamp: 1640995200,
            help: None,
        };
        
        let result = exporter.export(vec![sample]).await;
        assert!(result.is_ok());
        
        // Verify file was created and contains data
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("test_metric"));
        assert!(content.contains("3.14"));
    }
}