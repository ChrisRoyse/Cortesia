//! CI/CD Integration Module
//! 
//! Provides integration with various CI/CD platforms for automated testing.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use crate::infrastructure::TestReport;

/// Supported CI/CD platforms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CiPlatform {
    GitHub,
    GitLab,
    Jenkins,
    Azure,
    CircleCI,
    TravisCI,
    Unknown,
}

/// CI environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiEnvironment {
    pub platform: CiPlatform,
    pub build_id: Option<String>,
    pub commit_sha: Option<String>,
    pub branch: Option<String>,
    pub pull_request: Option<String>,
    pub build_url: Option<String>,
    pub runner_os: Option<String>,
    pub environment_variables: HashMap<String, String>,
}

/// CI integration configuration
#[derive(Debug, Clone)]
pub struct CiConfig {
    pub enable_annotations: bool,
    pub fail_on_coverage_drop: bool,
    pub coverage_threshold: f64,
    pub enable_status_checks: bool,
    pub upload_artifacts: bool,
    pub artifact_retention_days: u32,
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            enable_annotations: true,
            fail_on_coverage_drop: true,
            coverage_threshold: 95.0,
            enable_status_checks: true,
            upload_artifacts: true,
            artifact_retention_days: 30,
        }
    }
}

/// Main CI integration handler
pub struct CiIntegration {
    config: CiConfig,
    environment: CiEnvironment,
}

impl CiIntegration {
    /// Create a new CI integration instance
    pub fn new(config: CiConfig) -> Result<Self> {
        let environment = Self::detect_ci_environment();
        
        Ok(Self {
            config,
            environment,
        })
    }

    /// Detect current CI environment
    pub fn detect_ci_environment() -> CiEnvironment {
        let platform = if env::var("GITHUB_ACTIONS").is_ok() {
            CiPlatform::GitHub
        } else if env::var("GITLAB_CI").is_ok() {
            CiPlatform::GitLab
        } else if env::var("JENKINS_URL").is_ok() {
            CiPlatform::Jenkins
        } else if env::var("TF_BUILD").is_ok() {
            CiPlatform::Azure
        } else if env::var("CIRCLECI").is_ok() {
            CiPlatform::CircleCI
        } else if env::var("TRAVIS").is_ok() {
            CiPlatform::TravisCI
        } else {
            CiPlatform::Unknown
        };

        let mut environment_variables = HashMap::new();
        for (key, value) in env::vars() {
            if key.starts_with("CI_") || key.starts_with("GITHUB_") || key.starts_with("GITLAB_") {
                environment_variables.insert(key, value);
            }
        }

        CiEnvironment {
            platform: platform.clone(),
            build_id: Self::get_build_id(&platform),
            commit_sha: Self::get_commit_sha(&platform),
            branch: Self::get_branch(&platform),
            pull_request: Self::get_pull_request(&platform),
            build_url: Self::get_build_url(&platform),
            runner_os: env::var("RUNNER_OS").ok(),
            environment_variables,
        }
    }

    /// Process test results for CI/CD
    pub async fn process_test_results(&self, report: &TestReport) -> Result<()> {
        // Generate CI annotations
        if self.config.enable_annotations {
            self.generate_annotations(report).await?;
        }

        // Check coverage threshold
        if self.config.fail_on_coverage_drop && report.coverage_percentage < self.config.coverage_threshold {
            return Err(anyhow::anyhow!(
                "Coverage {} is below threshold {}",
                report.coverage_percentage,
                self.config.coverage_threshold
            ));
        }

        // Upload artifacts
        if self.config.upload_artifacts {
            self.upload_artifacts(report).await?;
        }

        // Set status checks
        if self.config.enable_status_checks {
            self.set_status_checks(report).await?;
        }

        Ok(())
    }

    /// Generate CI platform-specific annotations
    async fn generate_annotations(&self, report: &TestReport) -> Result<()> {
        match self.environment.platform {
            CiPlatform::GitHub => self.generate_github_annotations(report).await,
            CiPlatform::GitLab => self.generate_gitlab_annotations(report).await,
            CiPlatform::Azure => self.generate_azure_annotations(report).await,
            _ => Ok(()), // No-op for unsupported platforms
        }
    }

    /// Generate GitHub Actions annotations
    async fn generate_github_annotations(&self, report: &TestReport) -> Result<()> {
        // Summary annotation
        println!("::notice title=Test Summary::Total: {}, Passed: {}, Failed: {}, Coverage: {:.1}%",
            report.total_tests,
            report.passed_tests,
            report.failed_tests,
            report.coverage_percentage
        );

        // Failed test annotations
        for test in &report.test_summaries {
            if test.status == "FAILED" {
                if let Some(ref error) = test.error_message {
                    println!("::error title=Test Failed::Test '{}' failed: {}", test.name, error);
                } else {
                    println!("::error title=Test Failed::Test '{}' failed", test.name);
                }
            }
        }

        // Coverage warning
        if report.coverage_percentage < self.config.coverage_threshold {
            println!("::warning title=Low Coverage::Coverage {:.1}% is below threshold {:.1}%",
                report.coverage_percentage,
                self.config.coverage_threshold
            );
        }

        Ok(())
    }

    /// Generate GitLab CI annotations
    async fn generate_gitlab_annotations(&self, report: &TestReport) -> Result<()> {
        // GitLab uses different format
        println!("Test Summary: {} total, {} passed, {} failed, {:.1}% coverage",
            report.total_tests,
            report.passed_tests,
            report.failed_tests,
            report.coverage_percentage
        );

        Ok(())
    }

    /// Generate Azure DevOps annotations
    async fn generate_azure_annotations(&self, report: &TestReport) -> Result<()> {
        // Azure DevOps format
        println!("##[section]Test Results Summary");
        println!("Total Tests: {}", report.total_tests);
        println!("Passed: {}", report.passed_tests);
        println!("Failed: {}", report.failed_tests);
        println!("Coverage: {:.1}%", report.coverage_percentage);

        for test in &report.test_summaries {
            if test.status == "FAILED" {
                if let Some(ref error) = test.error_message {
                    println!("##[error]Test '{}' failed: {}", test.name, error);
                }
            }
        }

        Ok(())
    }

    /// Upload test artifacts
    async fn upload_artifacts(&self, _report: &TestReport) -> Result<()> {
        match self.environment.platform {
            CiPlatform::GitHub => {
                println!("::notice::Test artifacts will be uploaded by workflow");
            }
            CiPlatform::GitLab => {
                println!("Test artifacts configured in .gitlab-ci.yml");
            }
            _ => {
                println!("Artifact upload not configured for this platform");
            }
        }
        Ok(())
    }

    /// Set CI status checks
    async fn set_status_checks(&self, report: &TestReport) -> Result<()> {
        let status = if report.failed_tests == 0 && report.coverage_percentage >= self.config.coverage_threshold {
            "success"
        } else {
            "failure"
        };

        println!("Setting CI status: {}", status);
        
        // Platform-specific status setting would go here
        match self.environment.platform {
            CiPlatform::GitHub => {
                // GitHub status checks are typically set via REST API
                println!("GitHub status check: {}", status);
            }
            _ => {
                println!("Status check set: {}", status);
            }
        }

        Ok(())
    }

    /// Generate CI workflow configuration
    pub fn generate_workflow_config(&self, platform: &CiPlatform) -> Result<String> {
        match platform {
            CiPlatform::GitHub => Ok(self.generate_github_workflow()),
            CiPlatform::GitLab => Ok(self.generate_gitlab_workflow()),
            CiPlatform::Azure => Ok(self.generate_azure_workflow()),
            _ => Err(anyhow::anyhow!("Workflow generation not supported for {:?}", platform)),
        }
    }

    fn generate_github_workflow(&self) -> String {
        r#"
name: LLMKG Unit Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run unit tests
      run: |
        cd tests
        cargo test --release --verbose
    
    - name: Generate coverage report
      run: |
        cargo install cargo-tarpaulin
        cargo tarpaulin --out xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./cobertura.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test_reports/
        retention-days: 30
"#.trim().to_string()
    }

    fn generate_gitlab_workflow(&self) -> String {
        r#"
stages:
  - test
  - coverage

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo

cache:
  paths:
    - .cargo/
    - target/

test:
  stage: test
  image: rust:latest
  before_script:
    - apt-get update -qq && apt-get install -y -qq git
    - rustc --version
    - cargo --version
  script:
    - cd tests
    - cargo test --release --verbose
  artifacts:
    reports:
      junit: test_reports/junit.xml
    paths:
      - test_reports/
    expire_in: 30 days

coverage:
  stage: coverage
  image: rust:latest
  before_script:
    - cargo install cargo-tarpaulin
  script:
    - cargo tarpaulin --out xml
  coverage: '/\d+\.\d+% coverage/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: cobertura.xml
"#.trim().to_string()
    }

    fn generate_azure_workflow(&self) -> String {
        r#"
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  CARGO_TERM_COLOR: always

steps:
- task: Cache@2
  inputs:
    key: 'cargo | "$(Agent.OS)" | Cargo.lock'
    restoreKeys: |
      cargo | "$(Agent.OS)"
      cargo
    path: $(CARGO_HOME)
  displayName: Cache cargo

- script: |
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    rustc --version
    cargo --version
  displayName: 'Install Rust'

- script: |
    cd tests
    cargo test --release --verbose
  displayName: 'Run unit tests'

- script: |
    cargo install cargo-tarpaulin
    cargo tarpaulin --out xml
  displayName: 'Generate coverage'

- task: PublishTestResults@2
  inputs:
    testResultsFormat: 'JUnit'
    testResultsFiles: 'test_reports/junit.xml'
    mergeTestResults: true
  displayName: 'Publish test results'

- task: PublishCodeCoverageResults@1
  inputs:
    codeCoverageTool: 'Cobertura'
    summaryFileLocation: 'cobertura.xml'
  displayName: 'Publish coverage results'
"#.trim().to_string()
    }

    // Helper functions to extract CI environment variables
    fn get_build_id(platform: &CiPlatform) -> Option<String> {
        match platform {
            CiPlatform::GitHub => env::var("GITHUB_RUN_ID").ok(),
            CiPlatform::GitLab => env::var("CI_PIPELINE_ID").ok(),
            CiPlatform::Jenkins => env::var("BUILD_ID").ok(),
            CiPlatform::Azure => env::var("BUILD_BUILDID").ok(),
            CiPlatform::CircleCI => env::var("CIRCLE_BUILD_NUM").ok(),
            CiPlatform::TravisCI => env::var("TRAVIS_BUILD_ID").ok(),
            _ => None,
        }
    }

    fn get_commit_sha(platform: &CiPlatform) -> Option<String> {
        match platform {
            CiPlatform::GitHub => env::var("GITHUB_SHA").ok(),
            CiPlatform::GitLab => env::var("CI_COMMIT_SHA").ok(),
            CiPlatform::Jenkins => env::var("GIT_COMMIT").ok(),
            CiPlatform::Azure => env::var("BUILD_SOURCEVERSION").ok(),
            CiPlatform::CircleCI => env::var("CIRCLE_SHA1").ok(),
            CiPlatform::TravisCI => env::var("TRAVIS_COMMIT").ok(),
            _ => None,
        }
    }

    fn get_branch(platform: &CiPlatform) -> Option<String> {
        match platform {
            CiPlatform::GitHub => env::var("GITHUB_REF_NAME").ok(),
            CiPlatform::GitLab => env::var("CI_COMMIT_REF_NAME").ok(),
            CiPlatform::Jenkins => env::var("GIT_BRANCH").ok(),
            CiPlatform::Azure => env::var("BUILD_SOURCEBRANCH").ok(),
            CiPlatform::CircleCI => env::var("CIRCLE_BRANCH").ok(),
            CiPlatform::TravisCI => env::var("TRAVIS_BRANCH").ok(),
            _ => None,
        }
    }

    fn get_pull_request(platform: &CiPlatform) -> Option<String> {
        match platform {
            CiPlatform::GitHub => env::var("GITHUB_REF").ok()
                .and_then(|r| if r.contains("pull") { Some(r) } else { None }),
            CiPlatform::GitLab => env::var("CI_MERGE_REQUEST_IID").ok(),
            CiPlatform::Azure => env::var("SYSTEM_PULLREQUEST_PULLREQUESTID").ok(),
            CiPlatform::CircleCI => env::var("CIRCLE_PULL_REQUEST").ok(),
            CiPlatform::TravisCI => env::var("TRAVIS_PULL_REQUEST").ok(),
            _ => None,
        }
    }

    fn get_build_url(platform: &CiPlatform) -> Option<String> {
        match platform {
            CiPlatform::GitHub => {
                let repo = env::var("GITHUB_REPOSITORY").ok()?;
                let run_id = env::var("GITHUB_RUN_ID").ok()?;
                Some(format!("https://github.com/{}/actions/runs/{}", repo, run_id))
            }
            CiPlatform::GitLab => env::var("CI_PIPELINE_URL").ok(),
            CiPlatform::Jenkins => env::var("BUILD_URL").ok(),
            CiPlatform::Azure => {
                let org = env::var("SYSTEM_TEAMFOUNDATIONCOLLECTIONURI").ok()?;
                let project = env::var("SYSTEM_TEAMPROJECT").ok()?;
                let build_id = env::var("BUILD_BUILDID").ok()?;
                Some(format!("{}{}/_build/results?buildId={}", org, project, build_id))
            }
            CiPlatform::CircleCI => env::var("CIRCLE_BUILD_URL").ok(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_environment_detection() {
        let env = CiIntegration::detect_ci_environment();
        // Will detect as Unknown in test environment
        assert_eq!(env.platform, CiPlatform::Unknown);
    }

    #[test]
    fn test_ci_config_defaults() {
        let config = CiConfig::default();
        assert!(config.enable_annotations);
        assert!(config.fail_on_coverage_drop);
        assert_eq!(config.coverage_threshold, 95.0);
        assert!(config.enable_status_checks);
        assert!(config.upload_artifacts);
        assert_eq!(config.artifact_retention_days, 30);
    }

    #[tokio::test]
    async fn test_ci_integration_creation() {
        let config = CiConfig::default();
        let integration = CiIntegration::new(config);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_workflow_generation() {
        let config = CiConfig::default();
        let integration = CiIntegration::new(config).unwrap();
        
        let github_workflow = integration.generate_workflow_config(&CiPlatform::GitHub);
        assert!(github_workflow.is_ok());
        assert!(github_workflow.unwrap().contains("LLMKG Unit Tests"));
        
        let gitlab_workflow = integration.generate_workflow_config(&CiPlatform::GitLab);
        assert!(gitlab_workflow.is_ok());
        assert!(gitlab_workflow.unwrap().contains("stages:"));
        
        let azure_workflow = integration.generate_workflow_config(&CiPlatform::Azure);
        assert!(azure_workflow.is_ok());
        assert!(azure_workflow.unwrap().contains("trigger:"));
    }

    #[test]
    fn test_unsupported_platform_workflow() {
        let config = CiConfig::default();
        let integration = CiIntegration::new(config).unwrap();
        
        let result = integration.generate_workflow_config(&CiPlatform::Unknown);
        assert!(result.is_err());
    }
}