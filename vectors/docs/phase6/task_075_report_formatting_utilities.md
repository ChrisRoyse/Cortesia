# Task 075: Add Report Formatting Utilities

## Context
You are implementing report formatting utilities for a Rust-based vector indexing system. The ReportFormatter provides comprehensive formatting functions for tables, numbers, dates, and text across different output formats (markdown, HTML, JSON, CSV).

## Project Structure
```
src/
  validation/
    report_formatter.rs     <- Create this file
  lib.rs
```

## Task Description
Create the `ReportFormatter` struct that provides comprehensive formatting utilities for reports including table formatting, number formatting, date/time formatting, and text processing utilities.

## Requirements
1. Create `src/validation/report_formatter.rs`
2. Implement comprehensive formatting utilities for different data types
3. Add table formatting with alignment and styling options
4. Include number formatting with precision and units
5. Support multiple output formats (markdown, HTML, JSON, CSV)
6. Provide text processing utilities (truncation, wrapping, sanitization)

## Expected Code Structure
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use anyhow::{Result, Context};

#[derive(Debug, Clone)]
pub struct ReportFormatter {
    pub config: FormatterConfig,
    pub style_registry: StyleRegistry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatterConfig {
    pub default_format: OutputFormat,
    pub number_precision: usize,
    pub date_format: String,
    pub time_format: String,
    pub timezone: String,
    pub currency: String,
    pub locale: String,
    pub max_text_length: usize,
    pub table_style: TableStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Markdown,
    Html,
    Json,
    Csv,
    PlainText,
    Latex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TableStyle {
    Simple,
    Grid,
    Pipe,
    GitHub,
    Bootstrap,
    Professional,
}

#[derive(Debug, Clone)]
pub struct StyleRegistry {
    pub table_styles: HashMap<TableStyle, TableStyleDefinition>,
    pub text_styles: HashMap<String, TextStyle>,
    pub color_schemes: HashMap<String, ColorScheme>,
}

#[derive(Debug, Clone)]
pub struct TableStyleDefinition {
    pub header_separator: String,
    pub column_separator: String,
    pub row_prefix: String,
    pub row_suffix: String,
    pub alignment_chars: AlignmentChars,
    pub border_style: BorderStyle,
}

#[derive(Debug, Clone)]
pub struct AlignmentChars {
    pub left: String,
    pub center: String,
    pub right: String,
}

#[derive(Debug, Clone)]
pub struct BorderStyle {
    pub top: String,
    pub bottom: String,
    pub left: String,
    pub right: String,
    pub corner: String,
    pub intersection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStyle {
    pub font_weight: FontWeight,
    pub font_style: FontStyle,
    pub color: String,
    pub background_color: Option<String>,
    pub decoration: TextDecoration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextDecoration {
    None,
    Underline,
    Strikethrough,
    Overline,
}

#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub success: String,
    pub warning: String,
    pub danger: String,
    pub info: String,
    pub light: String,
    pub dark: String,
}

#[derive(Debug, Clone)]
pub struct FormattedTable {
    pub content: String,
    pub format: OutputFormat,
    pub metadata: TableMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    pub rows: usize,
    pub columns: usize,
    pub has_header: bool,
    pub total_width: usize,
    pub column_widths: Vec<usize>,
    pub alignments: Vec<Alignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Alignment {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Debug, Clone)]
pub struct TableData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub alignments: Vec<Alignment>,
    pub column_types: Vec<ColumnType>,
    pub footer: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnType {
    Text,
    Number,
    Percentage,
    Currency,
    Date,
    Time,
    DateTime,
    Boolean,
    Duration,
}

impl ReportFormatter {
    pub fn new() -> Self {
        Self {
            config: FormatterConfig::default(),
            style_registry: StyleRegistry::new(),
        }
    }

    pub fn with_config(config: FormatterConfig) -> Self {
        Self {
            config,
            style_registry: StyleRegistry::new(),
        }
    }

    // Table formatting methods
    pub fn format_table(&self, data: &TableData, format: OutputFormat) -> Result<FormattedTable> {
        match format {
            OutputFormat::Markdown => self.format_table_markdown(data),
            OutputFormat::Html => self.format_table_html(data),
            OutputFormat::Csv => self.format_table_csv(data),
            OutputFormat::Json => self.format_table_json(data),
            OutputFormat::PlainText => self.format_table_plain_text(data),
            OutputFormat::Latex => self.format_table_latex(data),
        }
    }

    fn format_table_markdown(&self, data: &TableData) -> Result<FormattedTable> {
        let mut content = String::new();
        let column_widths = self.calculate_column_widths(data);

        // Header row
        if !data.headers.is_empty() {
            content.push('|');
            for (i, header) in data.headers.iter().enumerate() {
                let width = column_widths[i];
                let formatted_header = self.pad_text(header, width, &data.alignments[i]);
                content.push_str(&format!(" {} |", formatted_header));
            }
            content.push('\n');

            // Separator row
            content.push('|');
            for (i, alignment) in data.alignments.iter().enumerate() {
                let width = column_widths[i];
                let separator = self.create_markdown_separator(width, alignment);
                content.push_str(&format!(" {} |", separator));
            }
            content.push('\n');
        }

        // Data rows
        for row in &data.rows {
            content.push('|');
            for (i, cell) in row.iter().enumerate() {
                let width = column_widths[i];
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                let padded_cell = self.pad_text(&formatted_cell, width, &data.alignments[i]);
                content.push_str(&format!(" {} |", padded_cell));
            }
            content.push('\n');
        }

        // Footer row (if present)
        if let Some(footer) = &data.footer {
            content.push('|');
            for (i, cell) in footer.iter().enumerate() {
                let width = column_widths[i];
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                let padded_cell = self.pad_text(&formatted_cell, width, &data.alignments[i]);
                content.push_str(&format!(" **{}** |", padded_cell));
            }
            content.push('\n');
        }

        Ok(FormattedTable {
            content,
            format: OutputFormat::Markdown,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width: column_widths.iter().sum::<usize>() + column_widths.len() * 3 + 1,
                column_widths,
                alignments: data.alignments.clone(),
            },
        })
    }

    fn format_table_html(&self, data: &TableData) -> Result<FormattedTable> {
        let mut content = String::from("<table class=\"report-table\">\n");

        // Header
        if !data.headers.is_empty() {
            content.push_str("  <thead>\n    <tr>\n");
            for (i, header) in data.headers.iter().enumerate() {
                let align_attr = match data.alignments[i] {
                    Alignment::Left => "left",
                    Alignment::Center => "center",
                    Alignment::Right => "right",
                    Alignment::Justify => "justify",
                };
                content.push_str(&format!("      <th align=\"{}\">{}</th>\n", align_attr, self.html_escape(header)));
            }
            content.push_str("    </tr>\n  </thead>\n");
        }

        // Body
        content.push_str("  <tbody>\n");
        for row in &data.rows {
            content.push_str("    <tr>\n");
            for (i, cell) in row.iter().enumerate() {
                let align_attr = match data.alignments[i] {
                    Alignment::Left => "left",
                    Alignment::Center => "center",
                    Alignment::Right => "right",
                    Alignment::Justify => "justify",
                };
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                content.push_str(&format!("      <td align=\"{}\">{}</td>\n", align_attr, self.html_escape(&formatted_cell)));
            }
            content.push_str("    </tr>\n");
        }
        content.push_str("  </tbody>\n");

        // Footer
        if let Some(footer) = &data.footer {
            content.push_str("  <tfoot>\n    <tr>\n");
            for (i, cell) in footer.iter().enumerate() {
                let align_attr = match data.alignments[i] {
                    Alignment::Left => "left",
                    Alignment::Center => "center",
                    Alignment::Right => "right",
                    Alignment::Justify => "justify",
                };
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                content.push_str(&format!("      <th align=\"{}\"><strong>{}</strong></th>\n", align_attr, self.html_escape(&formatted_cell)));
            }
            content.push_str("    </tr>\n  </tfoot>\n");
        }

        content.push_str("</table>");

        Ok(FormattedTable {
            content,
            format: OutputFormat::Html,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width: 0, // Not applicable for HTML
                column_widths: vec![0; data.headers.len()], // Not applicable for HTML
                alignments: data.alignments.clone(),
            },
        })
    }

    fn format_table_csv(&self, data: &TableData) -> Result<FormattedTable> {
        let mut content = String::new();

        // Header
        if !data.headers.is_empty() {
            let header_line = data.headers
                .iter()
                .map(|h| self.csv_escape(h))
                .collect::<Vec<_>>()
                .join(",");
            content.push_str(&header_line);
            content.push('\n');
        }

        // Data rows
        for row in &data.rows {
            let row_line = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                    self.csv_escape(&formatted_cell)
                })
                .collect::<Vec<_>>()
                .join(",");
            content.push_str(&row_line);
            content.push('\n');
        }

        Ok(FormattedTable {
            content,
            format: OutputFormat::Csv,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width: 0, // Not applicable for CSV
                column_widths: vec![0; data.headers.len()], // Not applicable for CSV
                alignments: data.alignments.clone(),
            },
        })
    }

    fn format_table_json(&self, data: &TableData) -> Result<FormattedTable> {
        let mut json_data = Vec::new();

        for row in &data.rows {
            let mut row_object = serde_json::Map::new();
            for (i, cell) in row.iter().enumerate() {
                let header = if i < data.headers.len() {
                    &data.headers[i]
                } else {
                    &format!("column_{}", i)
                };
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                row_object.insert(header.clone(), serde_json::Value::String(formatted_cell));
            }
            json_data.push(serde_json::Value::Object(row_object));
        }

        let content = serde_json::to_string_pretty(&json_data)
            .context("Failed to serialize table data to JSON")?;

        Ok(FormattedTable {
            content,
            format: OutputFormat::Json,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width: 0, // Not applicable for JSON
                column_widths: vec![0; data.headers.len()], // Not applicable for JSON
                alignments: data.alignments.clone(),
            },
        })
    }

    fn format_table_plain_text(&self, data: &TableData) -> Result<FormattedTable> {
        let mut content = String::new();
        let column_widths = self.calculate_column_widths(data);
        let total_width = column_widths.iter().sum::<usize>() + column_widths.len() * 3 + 1;
        let border = "+".to_string() + &"-".repeat(total_width - 2) + "+\n";

        content.push_str(&border);

        // Header
        if !data.headers.is_empty() {
            content.push('|');
            for (i, header) in data.headers.iter().enumerate() {
                let width = column_widths[i];
                let formatted_header = self.pad_text(header, width, &Alignment::Center);
                content.push_str(&format!(" {} |", formatted_header));
            }
            content.push('\n');
            content.push_str(&border);
        }

        // Data rows
        for row in &data.rows {
            content.push('|');
            for (i, cell) in row.iter().enumerate() {
                let width = column_widths[i];
                let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                let padded_cell = self.pad_text(&formatted_cell, width, &data.alignments[i]);
                content.push_str(&format!(" {} |", padded_cell));
            }
            content.push('\n');
        }

        content.push_str(&border);

        Ok(FormattedTable {
            content,
            format: OutputFormat::PlainText,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width,
                column_widths,
                alignments: data.alignments.clone(),
            },
        })
    }

    fn format_table_latex(&self, data: &TableData) -> Result<FormattedTable> {
        let mut content = String::new();
        
        // Table environment setup
        let column_spec = data.alignments
            .iter()
            .map(|align| match align {
                Alignment::Left => "l",
                Alignment::Center => "c",
                Alignment::Right => "r",
                Alignment::Justify => "p{3cm}",
            })
            .collect::<Vec<_>>()
            .join("");

        content.push_str(&format!("\\begin{{tabular}}{{{}}}\n", column_spec));
        content.push_str("\\hline\n");

        // Header
        if !data.headers.is_empty() {
            let header_line = data.headers
                .iter()
                .map(|h| self.latex_escape(h))
                .collect::<Vec<_>>()
                .join(" & ");
            content.push_str(&format!("\\textbf{{{}}} \\\\\n", header_line));
            content.push_str("\\hline\n");
        }

        // Data rows
        for row in &data.rows {
            let row_line = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                    self.latex_escape(&formatted_cell)
                })
                .collect::<Vec<_>>()
                .join(" & ");
            content.push_str(&format!("{} \\\\\n", row_line));
        }

        content.push_str("\\hline\n");
        content.push_str("\\end{tabular}");

        Ok(FormattedTable {
            content,
            format: OutputFormat::Latex,
            metadata: TableMetadata {
                rows: data.rows.len(),
                columns: data.headers.len(),
                has_header: !data.headers.is_empty(),
                total_width: 0, // Not applicable for LaTeX
                column_widths: vec![0; data.headers.len()], // Not applicable for LaTeX
                alignments: data.alignments.clone(),
            },
        })
    }

    // Number formatting methods
    pub fn format_number(&self, value: f64, precision: Option<usize>) -> String {
        let precision = precision.unwrap_or(self.config.number_precision);
        format!("{:.prec$}", value, prec = precision)
    }

    pub fn format_percentage(&self, value: f64, precision: Option<usize>) -> String {
        let precision = precision.unwrap_or(1);
        format!("{:.prec$}%", value, prec = precision)
    }

    pub fn format_currency(&self, value: f64, precision: Option<usize>) -> String {
        let precision = precision.unwrap_or(2);
        format!("{}{:.prec$}", self.config.currency, value, prec = precision)
    }

    pub fn format_file_size(&self, bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }

    pub fn format_duration(&self, duration: Duration) -> String {
        let total_seconds = duration.num_seconds();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}s", seconds)
        } else {
            format!("{}ms", duration.num_milliseconds())
        }
    }

    // Date and time formatting methods
    pub fn format_datetime(&self, dt: &DateTime<Utc>) -> String {
        dt.format(&format!("{} {}", self.config.date_format, self.config.time_format)).to_string()
    }

    pub fn format_date(&self, dt: &DateTime<Utc>) -> String {
        dt.format(&self.config.date_format).to_string()
    }

    pub fn format_time(&self, dt: &DateTime<Utc>) -> String {
        dt.format(&self.config.time_format).to_string()
    }

    pub fn format_relative_time(&self, dt: &DateTime<Utc>) -> String {
        let now = Utc::now();
        let diff = now.signed_duration_since(*dt);

        if diff.num_days() > 0 {
            format!("{} days ago", diff.num_days())
        } else if diff.num_hours() > 0 {
            format!("{} hours ago", diff.num_hours())
        } else if diff.num_minutes() > 0 {
            format!("{} minutes ago", diff.num_minutes())
        } else {
            "Just now".to_string()
        }
    }

    // Text formatting methods
    pub fn truncate_text(&self, text: &str, max_length: Option<usize>) -> String {
        let max_length = max_length.unwrap_or(self.config.max_text_length);
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length.saturating_sub(3)])
        }
    }

    pub fn wrap_text(&self, text: &str, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        for line in text.lines() {
            if line.len() <= width {
                lines.push(line.to_string());
            } else {
                let mut remaining = line;
                while remaining.len() > width {
                    let split_pos = remaining[..width]
                        .rfind(' ')
                        .unwrap_or(width);
                    lines.push(remaining[..split_pos].to_string());
                    remaining = &remaining[split_pos..].trim_start();
                }
                if !remaining.is_empty() {
                    lines.push(remaining.to_string());
                }
            }
        }
        lines
    }

    pub fn capitalize_words(&self, text: &str) -> String {
        text.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn sanitize_text(&self, text: &str) -> String {
        text.chars()
            .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
            .collect()
    }

    // Helper methods
    fn calculate_column_widths(&self, data: &TableData) -> Vec<usize> {
        let mut widths = vec![0; data.headers.len()];

        // Check header widths
        for (i, header) in data.headers.iter().enumerate() {
            widths[i] = widths[i].max(header.len());
        }

        // Check data row widths
        for row in &data.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                    widths[i] = widths[i].max(formatted_cell.len());
                }
            }
        }

        // Check footer widths
        if let Some(footer) = &data.footer {
            for (i, cell) in footer.iter().enumerate() {
                if i < widths.len() {
                    let formatted_cell = self.format_cell_value(cell, &data.column_types[i]);
                    widths[i] = widths[i].max(formatted_cell.len());
                }
            }
        }

        // Ensure minimum width
        for width in &mut widths {
            *width = (*width).max(3);
        }

        widths
    }

    fn pad_text(&self, text: &str, width: usize, alignment: &Alignment) -> String {
        if text.len() >= width {
            return text.to_string();
        }

        let padding = width - text.len();
        match alignment {
            Alignment::Left => format!("{}{}", text, " ".repeat(padding)),
            Alignment::Right => format!("{}{}", " ".repeat(padding), text),
            Alignment::Center => {
                let left_padding = padding / 2;
                let right_padding = padding - left_padding;
                format!("{}{}{}", " ".repeat(left_padding), text, " ".repeat(right_padding))
            }
            Alignment::Justify => text.to_string(), // For now, same as left
        }
    }

    fn create_markdown_separator(&self, width: usize, alignment: &Alignment) -> String {
        match alignment {
            Alignment::Left => format!(":{}", "-".repeat(width.saturating_sub(1))),
            Alignment::Right => format!("{}:", "-".repeat(width.saturating_sub(1))),
            Alignment::Center => format!(":{}:", "-".repeat(width.saturating_sub(2))),
            Alignment::Justify => "-".repeat(width),
        }
    }

    fn format_cell_value(&self, value: &str, column_type: &ColumnType) -> String {
        match column_type {
            ColumnType::Number => {
                if let Ok(num) = value.parse::<f64>() {
                    self.format_number(num, None)
                } else {
                    value.to_string()
                }
            }
            ColumnType::Percentage => {
                if let Ok(num) = value.parse::<f64>() {
                    self.format_percentage(num, None)
                } else {
                    value.to_string()
                }
            }
            ColumnType::Currency => {
                if let Ok(num) = value.parse::<f64>() {
                    self.format_currency(num, None)
                } else {
                    value.to_string()
                }
            }
            ColumnType::Date | ColumnType::Time | ColumnType::DateTime => {
                // For now, return as-is. In a real implementation, you'd parse and format
                value.to_string()
            }
            ColumnType::Boolean => {
                match value.to_lowercase().as_str() {
                    "true" | "1" | "yes" => "✅".to_string(),
                    "false" | "0" | "no" => "❌".to_string(),
                    _ => value.to_string(),
                }
            }
            ColumnType::Duration => {
                if let Ok(seconds) = value.parse::<i64>() {
                    self.format_duration(Duration::seconds(seconds))
                } else {
                    value.to_string()
                }
            }
            ColumnType::Text => value.to_string(),
        }
    }

    // Escape methods for different formats
    fn html_escape(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }

    fn csv_escape(&self, text: &str) -> String {
        if text.contains(',') || text.contains('"') || text.contains('\n') {
            format!("\"{}\"", text.replace('"', "\"\""))
        } else {
            text.to_string()
        }
    }

    fn latex_escape(&self, text: &str) -> String {
        text.replace('\\', "\\textbackslash{}")
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('$', "\\$")
            .replace('&', "\\&")
            .replace('%', "\\%")
            .replace('#', "\\#")
            .replace('^', "\\textasciicircum{}")
            .replace('_', "\\_")
            .replace('~', "\\textasciitilde{}")
    }
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            default_format: OutputFormat::Markdown,
            number_precision: 2,
            date_format: "%Y-%m-%d".to_string(),
            time_format: "%H:%M:%S".to_string(),
            timezone: "UTC".to_string(),
            currency: "$".to_string(),
            locale: "en_US".to_string(),
            max_text_length: 50,
            table_style: TableStyle::GitHub,
        }
    }
}

impl StyleRegistry {
    pub fn new() -> Self {
        let mut table_styles = HashMap::new();
        
        table_styles.insert(
            TableStyle::GitHub,
            TableStyleDefinition {
                header_separator: "-".to_string(),
                column_separator: "|".to_string(),
                row_prefix: "|".to_string(),
                row_suffix: "|".to_string(),
                alignment_chars: AlignmentChars {
                    left: ":--".to_string(),
                    center: ":-:".to_string(),
                    right: "--:".to_string(),
                },
                border_style: BorderStyle {
                    top: "".to_string(),
                    bottom: "".to_string(),
                    left: "|".to_string(),
                    right: "|".to_string(),
                    corner: "".to_string(),
                    intersection: "|".to_string(),
                },
            },
        );

        table_styles.insert(
            TableStyle::Grid,
            TableStyleDefinition {
                header_separator: "-".to_string(),
                column_separator: "|".to_string(),
                row_prefix: "|".to_string(),
                row_suffix: "|".to_string(),
                alignment_chars: AlignmentChars {
                    left: "---".to_string(),
                    center: "---".to_string(),
                    right: "---".to_string(),
                },
                border_style: BorderStyle {
                    top: "+".to_string(),
                    bottom: "+".to_string(),
                    left: "|".to_string(),
                    right: "|".to_string(),
                    corner: "+".to_string(),
                    intersection: "+".to_string(),
                },
            },
        );

        Self {
            table_styles,
            text_styles: HashMap::new(),
            color_schemes: HashMap::new(),
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            success: "#28a745".to_string(),
            warning: "#ffc107".to_string(),
            danger: "#dc3545".to_string(),
            info: "#17a2b8".to_string(),
            light: "#f8f9fa".to_string(),
            dark: "#343a40".to_string(),
        }
    }
}
```

## Dependencies to Add
```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
serde_json = "1.0"
```

## Success Criteria
- ReportFormatter compiles without errors
- All output formats (markdown, HTML, JSON, CSV) are supported
- Number, date, and text formatting functions work correctly
- Table formatting handles alignment and styling properly
- Text processing utilities handle edge cases
- Escaping functions prevent injection attacks

## Time Limit
10 minutes maximum