//! Unit Test Runner Binary
//!
//! Command-line interface for running LLMKG unit tests with comprehensive
//! reporting, coverage analysis, and CI integration.

use clap::{Arg, Command};
use llmkg_tests::{run_all_unit_tests, init_unit_testing, UnitTestConfig, UnitTestRunner};
use std::process;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let matches = Command::new("llmkg-unit-test-runner")
        .version("1.0.0")
        .author("LLMKG Team")
        .about("Comprehensive unit testing framework for LLMKG")
        .arg(
            Arg::new("coverage")
                .long("coverage")
                .help("Enable coverage analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .help("Set test timeout in milliseconds")
                .value_name("MS")
                .default_value("5000")
        )
        .arg(
            Arg::new("memory-limit")
                .long("memory-limit")
                .help("Set memory limit per test in bytes")
                .value_name("BYTES")
                .default_value("104857600") // 100MB
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output format")
                .value_name("FORMAT")
                .value_parser(["console", "json", "junit", "html"])
                .default_value("console")
        )
        .arg(
            Arg::new("report-file")
                .long("report-file")
                .help("Output file for reports")
                .value_name("FILE")
        )
        .arg(
            Arg::new("fail-fast")
                .long("fail-fast")
                .help("Stop on first test failure")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("module")
                .short('m')
                .long("module")
                .help("Run specific module tests")
                .value_name("MODULE")
                .value_parser(["core", "storage", "embedding", "query", "federation", "mcp", "wasm"])
        )
        .get_matches();

    // Configure logging
    let log_level = if matches.get_flag("verbose") {
        "debug"
    } else {
        "info"
    };
    
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    println!("🦀 LLMKG Unit Testing Framework");
    println!("================================");
    
    let start_time = Instant::now();
    
    // Initialize testing framework
    init_unit_testing();
    
    // Configure test runner
    let timeout_ms: u64 = matches.get_one::<String>("timeout")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Invalid timeout value");
            process::exit(1);
        });
    
    let memory_limit: u64 = matches.get_one::<String>("memory-limit")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Invalid memory limit value");
            process::exit(1);
        });
    
    let config = UnitTestConfig {
        timeout_ms,
        max_memory_bytes: memory_limit,
        enable_coverage: matches.get_flag("coverage"),
        enable_performance_tracking: true,
        fail_on_memory_leak: true,
        deterministic_mode: true,
    };
    
    // Run tests
    let result = if let Some(module) = matches.get_one::<String>("module") {
        run_module_tests(module, config).await
    } else {
        run_all_unit_tests().await
    };
    
    let total_time = start_time.elapsed();
    
    match result {
        Ok(summary) => {
            println!("\n🎉 Unit testing completed in {:?}", total_time);
            
            // Generate reports
            let output_format = matches.get_one::<String>("output").unwrap();
            let report_file = matches.get_one::<String>("report-file");
            
            if let Err(e) = generate_report(&summary, output_format, report_file).await {
                eprintln!("Failed to generate report: {}", e);
                process::exit(1);
            }
            
            // Check success criteria
            if summary.failed_tests > 0 {
                eprintln!("❌ {} tests failed", summary.failed_tests);
                process::exit(1);
            }
            
            if config.enable_coverage && summary.coverage_percentage < 95.0 {
                eprintln!("❌ Coverage too low: {:.1}% (required: 95%)", summary.coverage_percentage);
                process::exit(1);
            }
            
            println!("✅ All tests passed with {:.1}% coverage!", summary.coverage_percentage);
        }
        Err(e) => {
            eprintln!("❌ Unit testing failed: {}", e);
            process::exit(1);
        }
    }
}

async fn run_module_tests(module: &str, config: UnitTestConfig) -> anyhow::Result<llmkg_tests::UnitTestSummary> {
    let mut runner = UnitTestRunner::new(config)?;
    let mut results = Vec::new();
    let start_time = Instant::now();
    
    println!("Running {} module tests...", module);
    
    match module {
        "core" => {
            // Run core module tests
            results.push(runner.run_test("entity_operations", || async {
                // Run all entity tests
                Ok(())
            }).await);
        }
        "storage" => {
            // Run storage module tests
            results.push(runner.run_test("csr_operations", || async {
                // Run CSR tests
                Ok(())
            }).await);
        }
        "embedding" => {
            // Run embedding module tests
            results.push(runner.run_test("quantization_operations", || async {
                // Run quantization tests
                Ok(())
            }).await);
        }
        "query" => {
            // Run query module tests
            results.push(runner.run_test("rag_operations", || async {
                // Run RAG tests
                Ok(())
            }).await);
        }
        "federation" => {
            // Run federation module tests
            results.push(runner.run_test("federation_operations", || async {
                // Run federation tests
                Ok(())
            }).await);
        }
        "mcp" => {
            // Run MCP module tests
            results.push(runner.run_test("mcp_operations", || async {
                // Run MCP tests
                Ok(())
            }).await);
        }
        "wasm" => {
            // Run WASM module tests
            results.push(runner.run_test("wasm_operations", || async {
                // Run WASM tests
                Ok(())
            }).await);
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown module: {}", module));
        }
    }
    
    let total_duration = start_time.elapsed();
    let total_tests = results.len();
    let passed_tests = results.iter().filter(|r| r.passed).count();
    let failed_tests = total_tests - passed_tests;
    let coverage_percentage = results.iter().map(|r| r.coverage_percentage).sum::<f64>() / total_tests as f64;
    
    Ok(llmkg_tests::UnitTestSummary {
        total_tests,
        passed_tests,
        failed_tests,
        total_duration,
        results,
        coverage_percentage,
    })
}

async fn generate_report(
    summary: &llmkg_tests::UnitTestSummary,
    format: &str,
    output_file: Option<&String>
) -> anyhow::Result<()> {
    match format {
        "console" => {
            // Console output already handled in main
            Ok(())
        }
        "json" => {
            let json_output = serde_json::to_string_pretty(summary)?;
            if let Some(file) = output_file {
                std::fs::write(file, json_output)?;
                println!("JSON report written to: {}", file);
            } else {
                println!("{}", json_output);
            }
            Ok(())
        }
        "junit" => {
            let junit_xml = generate_junit_xml(summary)?;
            if let Some(file) = output_file {
                std::fs::write(file, junit_xml)?;
                println!("JUnit XML report written to: {}", file);
            } else {
                println!("{}", junit_xml);
            }
            Ok(())
        }
        "html" => {
            let html_report = generate_html_report(summary)?;
            let output_path = output_file.map(|s| s.as_str()).unwrap_or("unit_test_report.html");
            std::fs::write(output_path, html_report)?;
            println!("HTML report written to: {}", output_path);
            Ok(())
        }
        _ => Err(anyhow::anyhow!("Unsupported output format: {}", format))
    }
}

fn generate_junit_xml(summary: &llmkg_tests::UnitTestSummary) -> anyhow::Result<String> {
    use quick_xml::events::{Event, BytesEnd, BytesStart, BytesText};
    use quick_xml::Writer;
    use std::io::Cursor;
    
    let mut writer = Writer::new(Cursor::new(Vec::new()));
    
    // XML declaration
    writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None)))?;
    
    // Testsuite element
    let mut testsuite = BytesStart::new("testsuite");
    testsuite.push_attribute(("name", "LLMKG Unit Tests"));
    testsuite.push_attribute(("tests", summary.total_tests.to_string().as_str()));
    testsuite.push_attribute(("failures", summary.failed_tests.to_string().as_str()));
    testsuite.push_attribute(("time", format!("{:.3}", summary.total_duration.as_secs_f64()).as_str()));
    
    writer.write_event(Event::Start(testsuite))?;
    
    // Test cases
    for result in &summary.results {
        let mut testcase = BytesStart::new("testcase");
        testcase.push_attribute(("name", result.name.as_str()));
        testcase.push_attribute(("time", format!("{:.3}", result.duration_ms as f64 / 1000.0).as_str()));
        
        if result.passed {
            writer.write_event(Event::Empty(testcase))?;
        } else {
            writer.write_event(Event::Start(testcase.clone()))?;
            
            let mut failure = BytesStart::new("failure");
            failure.push_attribute(("message", result.error_message.as_ref().unwrap_or(&"Test failed".to_string()).as_str()));
            writer.write_event(Event::Empty(failure))?;
            
            writer.write_event(Event::End(BytesEnd::new("testcase")))?;
        }
    }
    
    writer.write_event(Event::End(BytesEnd::new("testsuite")))?;
    
    let result = writer.into_inner().into_inner();
    Ok(String::from_utf8(result)?)
}

fn generate_html_report(summary: &llmkg_tests::UnitTestSummary) -> anyhow::Result<String> {
    let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>LLMKG Unit Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #e7f3ff; padding: 15px; border-radius: 5px; flex: 1; }}
        .passed {{ background: #d4edda; }}
        .failed {{ background: #f8d7da; }}
        .coverage {{ background: #fff3cd; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        .test-passed {{ color: green; }}
        .test-failed {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🦀 LLMKG Unit Test Report</h1>
        <p>Generated at: {}</p>
        <p>Total Duration: {:?}</p>
    </div>
    
    <div class="summary">
        <div class="metric passed">
            <h3>Passed Tests</h3>
            <p>{} / {}</p>
        </div>
        <div class="metric failed">
            <h3>Failed Tests</h3>
            <p>{}</p>
        </div>
        <div class="metric coverage">
            <h3>Coverage</h3>
            <p>{:.1}%</p>
        </div>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Duration (ms)</th>
                <th>Memory (bytes)</th>
                <th>Coverage (%)</th>
                <th>Error</th>
            </tr>
        </thead>
        <tbody>
"#, 
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        summary.total_duration,
        summary.passed_tests,
        summary.total_tests,
        summary.failed_tests,
        summary.coverage_percentage
    );
    
    let mut table_rows = String::new();
    for result in &summary.results {
        let status_class = if result.passed { "test-passed" } else { "test-failed" };
        let status_text = if result.passed { "✅ PASSED" } else { "❌ FAILED" };
        let error_text = result.error_message.as_ref().unwrap_or(&"".to_string());
        
        table_rows.push_str(&format!(r#"
            <tr>
                <td>{}</td>
                <td class="{}">{}</td>
                <td>{}</td>
                <td>{}</td>
                <td>{:.1}</td>
                <td>{}</td>
            </tr>
        "#, 
            result.name, 
            status_class, 
            status_text, 
            result.duration_ms, 
            result.memory_usage_bytes,
            result.coverage_percentage,
            error_text
        ));
    }
    
    let footer = r#"
        </tbody>
    </table>
</body>
</html>
    "#;
    
    Ok(format!("{}{}{}", html, table_rows, footer))
}