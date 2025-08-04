# Task 076: Create Report Serialization System

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 069 (ValidationReport). The report serialization system enables export to multiple formats for different stakeholders and monitoring systems.

## Project Structure
```
src/
  validation/
    report.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement comprehensive report serialization that exports validation reports to multiple formats (JSON, CSV, HTML, XML) with schema validation and streaming capabilities for large reports.

## Requirements
1. Multi-format export (JSON, CSV, HTML, XML)
2. Schema validation for all export formats
3. Streaming export for large reports
4. Data compression and optimization
5. Integration with monitoring systems (Prometheus, InfluxDB)

## Expected Code Structure to Add
```rust
use serde::{Deserialize, Serialize};
use std::io::{Write, BufWriter};
use std::path::Path;
use anyhow::{Result, Context};

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    JsonPretty,
    Csv,
    Html,
    Xml,
    Prometheus,
    InfluxDb,
}

pub struct ReportSerializer {
    compression_enabled: bool,
    streaming_threshold_mb: usize,
    include_raw_data: bool,
}

impl ReportSerializer {
    pub fn new() -> Self {
        Self {
            compression_enabled: true,
            streaming_threshold_mb: 10,
            include_raw_data: false,
        }
    }
    
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }
    
    pub fn with_streaming_threshold(mut self, threshold_mb: usize) -> Self {
        self.streaming_threshold_mb = threshold_mb;
        self
    }
    
    pub fn serialize_report<P: AsRef<Path>>(
        &self,
        report: &ValidationReport,
        format: ExportFormat,
        output_path: P,
    ) -> Result<SerializationResult> {
        let output_path = output_path.as_ref();
        
        // Estimate report size
        let estimated_size = self.estimate_report_size(report)?;
        let use_streaming = estimated_size > (self.streaming_threshold_mb * 1_048_576);
        
        let result = if use_streaming {
            self.serialize_streaming(report, format, output_path)?
        } else {
            self.serialize_buffered(report, format, output_path)?
        };
        
        // Validate output
        self.validate_output(output_path, &format)?;
        
        Ok(result)
    }
    
    fn serialize_buffered(
        &self,
        report: &ValidationReport,
        format: ExportFormat,
        output_path: &Path,
    ) -> Result<SerializationResult> {
        let start_time = std::time::Instant::now();
        
        let content = match format {
            ExportFormat::Json => {
                serde_json::to_string(report)
                    .context("Failed to serialize report to JSON")?
            }
            ExportFormat::JsonPretty => {
                serde_json::to_string_pretty(report)
                    .context("Failed to serialize report to pretty JSON")?
            }
            ExportFormat::Csv => {
                self.serialize_to_csv(report)?
            }
            ExportFormat::Html => {
                self.serialize_to_html(report)?
            }
            ExportFormat::Xml => {
                self.serialize_to_xml(report)?
            }
            ExportFormat::Prometheus => {
                self.serialize_to_prometheus(report)?
            }
            ExportFormat::InfluxDb => {
                self.serialize_to_influxdb(report)?
            }
        };
        
        // Write to file with optional compression
        let final_content = if self.compression_enabled {
            self.compress_content(&content)?
        } else {
            content.into_bytes()
        };
        
        std::fs::write(output_path, final_content)
            .with_context(|| format!("Failed to write report to {}", output_path.display()))?;
        
        let duration = start_time.elapsed();
        let file_size = std::fs::metadata(output_path)?.len();
        
        Ok(SerializationResult {
            format,
            file_size_bytes: file_size,
            serialization_duration: duration,
            compression_ratio: if self.compression_enabled {
                Some(content.len() as f64 / file_size as f64)
            } else {
                None
            },
            validation_passed: true,
        })
    }
    
    fn serialize_streaming(
        &self,
        report: &ValidationReport,
        format: ExportFormat,
        output_path: &Path,
    ) -> Result<SerializationResult> {
        let start_time = std::time::Instant::now();
        let file = std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path.display()))?;
        let mut writer = BufWriter::new(file);
        
        match format {
            ExportFormat::Json | ExportFormat::JsonPretty => {
                self.stream_json(report, &mut writer, format == ExportFormat::JsonPretty)?;
            }
            ExportFormat::Csv => {
                self.stream_csv(report, &mut writer)?;
            }
            ExportFormat::Html => {
                self.stream_html(report, &mut writer)?;
            }
            _ => {
                return Err(anyhow::anyhow!("Streaming not supported for format: {:?}", format));
            }
        }
        
        writer.flush()?;
        let duration = start_time.elapsed();
        let file_size = std::fs::metadata(output_path)?.len();
        
        Ok(SerializationResult {
            format,
            file_size_bytes: file_size,
            serialization_duration: duration,
            compression_ratio: None,
            validation_passed: true,
        })
    }
    
    fn serialize_to_csv(&self, report: &ValidationReport) -> Result<String> {
        let mut csv_content = String::new();
        
        // Header
        csv_content.push_str("Metric,Value,Category,Timestamp\n");
        
        // Overall metrics
        csv_content.push_str(&format!(
            "Overall Score,{:.2},Summary,{}\n",
            report.overall_score,
            report.metadata.generated_at.to_rfc3339()
        ));
        
        csv_content.push_str(&format!(
            "Overall Accuracy,{:.2},Accuracy,{}\n",
            report.accuracy_metrics.overall_accuracy,
            report.metadata.generated_at.to_rfc3339()
        ));
        
        // Query type breakdown
        for (query_type, result) in &report.accuracy_metrics.query_type_results {
            csv_content.push_str(&format!(
                "{} Accuracy,{:.2},Query Types,{}\n",
                query_type,
                result.accuracy_percentage,
                report.metadata.generated_at.to_rfc3339()
            ));
        }
        
        // Performance metrics
        csv_content.push_str(&format!(
            "P50 Latency,{},Performance,{}\n",
            report.performance_metrics.latency_metrics.p50_latency_ms,
            report.metadata.generated_at.to_rfc3339()
        ));
        
        csv_content.push_str(&format!(
            "Throughput QPS,{:.2},Performance,{}\n",
            report.performance_metrics.throughput_metrics.queries_per_second,
            report.metadata.generated_at.to_rfc3339()
        ));
        
        Ok(csv_content)
    }
    
    fn serialize_to_html(&self, report: &ValidationReport) -> Result<String> {
        let html_template = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMKG Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }}
        .success {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .warning {{ background-color: #fff3cd; border-color: #ffeaa7; }}
        .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLMKG Vector Indexing System - Validation Report</h1>
        <p><strong>Generated:</strong> {generated_at}</p>
        <p><strong>Overall Score:</strong> {overall_score:.1}/100</p>
    </div>
    
    <h2>Summary Metrics</h2>
    <div class="metric {accuracy_class}">
        <h3>Accuracy</h3>
        <p>{overall_accuracy:.1}%</p>
    </div>
    <div class="metric {performance_class}">
        <h3>Performance</h3>
        <p>P50: {p50_latency}ms</p>
    </div>
    <div class="metric {throughput_class}">
        <h3>Throughput</h3>
        <p>{throughput:.1} QPS</p>
    </div>
    
    <h2>Query Type Results</h2>
    <table>
        <thead>
            <tr>
                <th>Query Type</th>
                <th>Test Cases</th>
                <th>Passed</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
            </tr>
        </thead>
        <tbody>
            {query_type_rows}
        </tbody>
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        {recommendations}
    </ul>
</body>
</html>
        "#;
        
        let accuracy_class = if report.accuracy_metrics.overall_accuracy >= 95.0 {
            "success"
        } else if report.accuracy_metrics.overall_accuracy >= 80.0 {
            "warning"
        } else {
            "error"
        };
        
        let performance_class = if report.performance_metrics.meets_targets {
            "success"
        } else {
            "warning"
        };
        
        let throughput_class = if report.performance_metrics.throughput_metrics.queries_per_second >= 100.0 {
            "success"
        } else {
            "warning"
        };
        
        let query_type_rows = report.accuracy_metrics.query_type_results
            .iter()
            .map(|(query_type, result)| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                    format!("{:?}", query_type),
                    result.test_cases_count,
                    result.passed_count,
                    result.accuracy_percentage,
                    result.average_precision,
                    result.average_recall,
                    result.average_f1_score
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        
        let recommendations = report.recommendations
            .iter()
            .map(|rec| format!("<li>{}</li>", rec))
            .collect::<Vec<_>>()
            .join("\n");
        
        Ok(html_template
            .replace("{generated_at}", &report.metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .replace("{overall_score}", &format!("{:.1}", report.overall_score))
            .replace("{accuracy_class}", accuracy_class)
            .replace("{overall_accuracy}", &format!("{:.1}", report.accuracy_metrics.overall_accuracy))
            .replace("{performance_class}", performance_class)
            .replace("{p50_latency}", &report.performance_metrics.latency_metrics.p50_latency_ms.to_string())
            .replace("{throughput_class}", throughput_class)
            .replace("{throughput}", &format!("{:.1}", report.performance_metrics.throughput_metrics.queries_per_second))
            .replace("{query_type_rows}", &query_type_rows)
            .replace("{recommendations}", &recommendations))
    }
    
    fn serialize_to_prometheus(&self, report: &ValidationReport) -> Result<String> {
        let mut prometheus_metrics = String::new();
        
        // Overall metrics
        prometheus_metrics.push_str(&format!(
            "# HELP llmkg_overall_score Overall validation score\n# TYPE llmkg_overall_score gauge\nllmkg_overall_score {:.2}\n\n",
            report.overall_score
        ));
        
        prometheus_metrics.push_str(&format!(
            "# HELP llmkg_accuracy_overall Overall accuracy percentage\n# TYPE llmkg_accuracy_overall gauge\nllmkg_accuracy_overall {:.2}\n\n",
            report.accuracy_metrics.overall_accuracy
        ));
        
        // Query type metrics
        for (query_type, result) in &report.accuracy_metrics.query_type_results {
            prometheus_metrics.push_str(&format!(
                "llmkg_accuracy_by_type{{query_type=\"{:?}\"}} {:.2}\n",
                query_type, result.accuracy_percentage
            ));
        }
        
        // Performance metrics
        prometheus_metrics.push_str(&format!(
            "\n# HELP llmkg_latency_p50 P50 latency in milliseconds\n# TYPE llmkg_latency_p50 gauge\nllmkg_latency_p50 {}\n\n",
            report.performance_metrics.latency_metrics.p50_latency_ms
        ));
        
        prometheus_metrics.push_str(&format!(
            "# HELP llmkg_throughput_qps Queries per second\n# TYPE llmkg_throughput_qps gauge\nllmkg_throughput_qps {:.2}\n\n",
            report.performance_metrics.throughput_metrics.queries_per_second
        ));
        
        Ok(prometheus_metrics)
    }
    
    fn compress_content(&self, content: &str) -> Result<Vec<u8>> {
        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(content.as_bytes())?;
        Ok(encoder.finish()?)
    }
    
    fn validate_output(&self, output_path: &Path, format: &ExportFormat) -> Result<()> {
        let content = std::fs::read_to_string(output_path)?;
        
        match format {
            ExportFormat::Json | ExportFormat::JsonPretty => {
                serde_json::from_str::<serde_json::Value>(&content)
                    .context("Invalid JSON in output file")?;
            }
            ExportFormat::Html => {
                if !content.contains("<!DOCTYPE html>") || !content.contains("</html>") {
                    return Err(anyhow::anyhow!("Invalid HTML structure"));
                }
            }
            _ => {
                // Basic validation for other formats
                if content.is_empty() {
                    return Err(anyhow::anyhow!("Output file is empty"));
                }
            }
        }
        
        Ok(())
    }
    
    fn estimate_report_size(&self, report: &ValidationReport) -> Result<usize> {
        // Rough estimation based on report structure
        let base_size = std::mem::size_of::<ValidationReport>();
        let query_results_size = report.accuracy_metrics.query_type_results.len() * 1024;
        let recommendations_size = report.recommendations.iter().map(|r| r.len()).sum::<usize>();
        
        Ok(base_size + query_results_size + recommendations_size)
    }
}

#[derive(Debug, Clone)]
pub struct SerializationResult {
    pub format: ExportFormat,
    pub file_size_bytes: u64,
    pub serialization_duration: std::time::Duration,
    pub compression_ratio: Option<f64>,
    pub validation_passed: bool,
}

impl ValidationReport {
    pub fn export<P: AsRef<Path>>(
        &self,
        format: ExportFormat,
        output_path: P,
    ) -> Result<SerializationResult> {
        let serializer = ReportSerializer::new();
        serializer.serialize_report(self, format, output_path)
    }
    
    pub fn export_all<P: AsRef<Path>>(
        &self,
        output_dir: P,
        formats: Vec<ExportFormat>,
    ) -> Result<Vec<SerializationResult>> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        
        let mut results = Vec::new();
        let serializer = ReportSerializer::new();
        
        for format in formats {
            let filename = match format {
                ExportFormat::Json => "validation_report.json",
                ExportFormat::JsonPretty => "validation_report_pretty.json",
                ExportFormat::Csv => "validation_report.csv",
                ExportFormat::Html => "validation_report.html",
                ExportFormat::Xml => "validation_report.xml",
                ExportFormat::Prometheus => "validation_metrics.prom",
                ExportFormat::InfluxDb => "validation_metrics.influx",
            };
            
            let output_path = output_dir.join(filename);
            let result = serializer.serialize_report(self, format, output_path)?;
            results.push(result);
        }
        
        Ok(results)
    }
}
```

## Success Criteria
- Multi-format export works correctly for all supported formats
- Streaming works for large reports without memory issues
- Schema validation catches malformed outputs
- Compression reduces file sizes significantly
- Integration with monitoring systems produces valid metrics

## Time Limit
10 minutes maximum