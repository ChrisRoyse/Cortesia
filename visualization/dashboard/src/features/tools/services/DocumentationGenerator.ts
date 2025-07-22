import { MCPTool, JSONSchema, ToolDocumentation, ParameterDoc, ReturnDoc, CodeExample, ToolExample } from '../types';

interface ApiReference {
  endpoint: string;
  method: string;
  headers?: Record<string, string>;
  requestBody?: JSONSchema;
  responseBody?: JSONSchema;
  authentication?: string;
  rateLimit?: string;
}

export class DocumentationGenerator {
  /**
   * Generate complete documentation for a tool
   */
  generateDocumentation(tool: MCPTool): ToolDocumentation {
    const parameters = this.extractParameters(tool.inputSchema);
    const returns = this.extractReturns(tool.outputSchema);
    const examples = this.generateCodeExamples(tool);

    return {
      summary: this.generateSummary(tool),
      description: this.generateDetailedDescription(tool),
      parameters,
      returns,
      examples,
      relatedTools: this.findRelatedTools(tool),
      tags: tool.tags || []
    };
  }

  /**
   * Generate code examples in multiple languages
   */
  generateCodeExamples(tool: MCPTool): CodeExample[] {
    const examples: CodeExample[] = [];

    // JavaScript/TypeScript example
    examples.push({
      language: 'javascript',
      code: this.generateJavaScriptExample(tool),
      description: 'Using the tool with JavaScript/TypeScript'
    });

    // Python example
    examples.push({
      language: 'python',
      code: this.generatePythonExample(tool),
      description: 'Using the tool with Python'
    });

    // cURL example
    examples.push({
      language: 'curl',
      code: this.generateCurlExample(tool),
      description: 'Using the tool with cURL'
    });

    // Rust example
    examples.push({
      language: 'rust',
      code: this.generateRustExample(tool),
      description: 'Using the tool with Rust'
    });

    return examples;
  }

  /**
   * Generate API reference documentation
   */
  generateApiReference(tool: MCPTool): ApiReference {
    return {
      endpoint: tool.endpoint || `/api/tools/${tool.id}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Tool-Version': tool.version
      },
      requestBody: tool.inputSchema,
      responseBody: tool.outputSchema,
      authentication: 'Bearer token required',
      rateLimit: '100 requests per minute'
    };
  }

  /**
   * Export documentation in various formats
   */
  exportDocumentation(tools: MCPTool[], format: 'markdown' | 'html' | 'pdf'): string {
    switch (format) {
      case 'markdown':
        return this.exportMarkdown(tools);
      case 'html':
        return this.exportHtml(tools);
      case 'pdf':
        return this.exportPdf(tools);
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  // Private helper methods

  private generateSummary(tool: MCPTool): string {
    const categoryMap: Record<string, string> = {
      'knowledge-graph': 'Knowledge Graph',
      'cognitive': 'Cognitive Processing',
      'neural': 'Neural Network',
      'memory': 'Memory Operations',
      'analysis': 'Data Analysis',
      'federation': 'Federation',
      'utility': 'Utility'
    };

    return `${categoryMap[tool.category] || tool.category} tool for ${tool.description.toLowerCase()}`;
  }

  private generateDetailedDescription(tool: MCPTool): string {
    let description = tool.description;

    // Add category-specific details
    switch (tool.category) {
      case 'knowledge-graph':
        description += '\n\nThis tool operates on LLMKG\'s knowledge graph structure, allowing you to query, manipulate, and analyze entity relationships and semantic connections.';
        break;
      case 'cognitive':
        description += '\n\nPart of LLMKG\'s cognitive processing suite, this tool helps with pattern recognition, learning, and advanced reasoning capabilities.';
        break;
      case 'neural':
        description += '\n\nThis tool interfaces with LLMKG\'s neural network components, providing access to neural activity data and network state information.';
        break;
      case 'memory':
        description += '\n\nMemory management tool that handles storage, retrieval, and optimization of LLMKG\'s hierarchical memory systems.';
        break;
    }

    // Add performance characteristics
    if (tool.responseTime) {
      description += `\n\n**Performance**: Average response time of ${tool.responseTime}ms.`;
    }

    return description;
  }

  private extractParameters(schema: JSONSchema): ParameterDoc[] {
    const parameters: ParameterDoc[] = [];
    
    if (schema.properties) {
      Object.entries(schema.properties).forEach(([name, prop]: [string, any]) => {
        parameters.push({
          name,
          type: this.getTypeString(prop),
          description: prop.description || 'No description available',
          required: schema.required?.includes(name) || false,
          default: prop.default,
          examples: prop.examples || this.generateExamplesForType(prop)
        });
      });
    }

    return parameters;
  }

  private extractReturns(schema?: JSONSchema): ReturnDoc {
    if (!schema) {
      return {
        type: 'void',
        description: 'No return value'
      };
    }

    return {
      type: this.getTypeString(schema),
      description: schema.description || 'Returns the execution result',
      schema
    };
  }

  private getTypeString(prop: any): string {
    if (prop.type === 'array') {
      return `${prop.items?.type || 'any'}[]`;
    }
    if (prop.type === 'object') {
      if (prop.properties) {
        const props = Object.keys(prop.properties).join(', ');
        return `{ ${props} }`;
      }
      return 'object';
    }
    if (prop.enum) {
      return prop.enum.map((v: any) => `'${v}'`).join(' | ');
    }
    return prop.type || 'any';
  }

  private generateExamplesForType(prop: any): any[] {
    const examples: any[] = [];

    switch (prop.type) {
      case 'string':
        if (prop.enum) {
          examples.push(...prop.enum);
        } else {
          examples.push('example-value', 'another-example');
        }
        break;
      case 'number':
        examples.push(42, 3.14, -1);
        break;
      case 'boolean':
        examples.push(true, false);
        break;
      case 'array':
        examples.push([], ['item1', 'item2']);
        break;
      case 'object':
        examples.push({}, { key: 'value' });
        break;
    }

    return examples;
  }

  private generateJavaScriptExample(tool: MCPTool): string {
    const params = this.generateExampleParams(tool);
    
    return `// Import the LLMKG client
import { LLMKGClient } from '@llmkg/client';

// Initialize the client
const client = new LLMKGClient({
  apiKey: process.env.LLMKG_API_KEY
});

// Execute the ${tool.name} tool
async function use${this.toPascalCase(tool.name)}() {
  try {
    const result = await client.tools.execute('${tool.id}', ${JSON.stringify(params, null, 2)});
    
    console.log('Result:', result);
    return result;
  } catch (error) {
    console.error('Error executing tool:', error);
    throw error;
  }
}

// Error handling example
use${this.toPascalCase(tool.name)}()
  .then(result => {
    // Process the result
    if (result.success) {
      console.log('Tool executed successfully');
    }
  })
  .catch(error => {
    // Handle specific error types
    if (error.code === 'RATE_LIMIT') {
      console.log('Rate limit exceeded, retry after:', error.retryAfter);
    }
  });`;
  }

  private generatePythonExample(tool: MCPTool): string {
    const params = this.generateExampleParams(tool);
    
    return `# Import the LLMKG client
from llmkg import LLMKGClient
import os

# Initialize the client
client = LLMKGClient(
    api_key=os.environ.get('LLMKG_API_KEY')
)

# Execute the ${tool.name} tool
def use_${this.toSnakeCase(tool.name)}():
    try:
        result = client.tools.execute(
            tool_id='${tool.id}',
            params=${JSON.stringify(params, null, 12).replace(/^/gm, ' '.repeat(12)).trim()}
        )
        
        print(f"Result: {result}")
        return result
    except Exception as e:
        print(f"Error executing tool: {e}")
        raise

# Async example with error handling
import asyncio

async def use_${this.toSnakeCase(tool.name)}_async():
    try:
        async with client.async_session() as session:
            result = await session.tools.execute('${tool.id}', ${JSON.stringify(params)})
            return result
    except RateLimitError as e:
        print(f"Rate limit exceeded, retry after: {e.retry_after}")
    except ToolExecutionError as e:
        print(f"Tool execution failed: {e.message}")

# Run the async example
if __name__ == "__main__":
    result = asyncio.run(use_${this.toSnakeCase(tool.name)}_async())`;
  }

  private generateCurlExample(tool: MCPTool): string {
    const params = this.generateExampleParams(tool);
    const endpoint = tool.endpoint || `/api/tools/${tool.id}/execute`;
    
    return `# Execute ${tool.name} using cURL
curl -X POST https://api.llmkg.com${endpoint} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $LLMKG_API_KEY" \\
  -H "X-Tool-Version: ${tool.version}" \\
  -d '${JSON.stringify({ toolId: tool.id, params }, null, 2)}'

# Pretty print the response
curl -X POST https://api.llmkg.com${endpoint} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $LLMKG_API_KEY" \\
  -H "X-Tool-Version: ${tool.version}" \\
  -d '${JSON.stringify({ toolId: tool.id, params }, null, 2)}' \\
  | jq '.'

# Save response to file
curl -X POST https://api.llmkg.com${endpoint} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $LLMKG_API_KEY" \\
  -H "X-Tool-Version: ${tool.version}" \\
  -d '${JSON.stringify({ toolId: tool.id, params }, null, 2)}' \\
  -o response.json`;
  }

  private generateRustExample(tool: MCPTool): string {
    const params = this.generateExampleParams(tool);
    
    return `use llmkg_client::{Client, ToolExecutor, Error};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize the client
    let client = Client::new()
        .with_api_key(std::env::var("LLMKG_API_KEY")?)
        .build()?;

    // Execute the ${tool.name} tool
    let result = client
        .tools()
        .execute("${tool.id}")
        .with_params(json!(${JSON.stringify(params, null, 8).replace(/^/gm, ' '.repeat(8)).trim()}))
        .send()
        .await?;

    println!("Result: {:?}", result);

    // Error handling example
    match client.tools().execute("${tool.id}").with_params(json!(${JSON.stringify(params)})).send().await {
        Ok(result) => {
            println!("Success: {:?}", result);
        }
        Err(Error::RateLimit { retry_after }) => {
            println!("Rate limited. Retry after {} seconds", retry_after);
        }
        Err(Error::ToolExecution { message, code }) => {
            println!("Tool execution error ({}): {}", code, message);
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
        }
    }

    Ok(())
}

// Streaming example for tools that support it
async fn stream_example(client: &Client) -> Result<(), Error> {
    use futures::StreamExt;

    let mut stream = client
        .tools()
        .execute_stream("${tool.id}")
        .with_params(json!(${JSON.stringify(params)}))
        .send()
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            ToolEvent::Progress(p) => println!("Progress: {}%", p),
            ToolEvent::Data(d) => println!("Data: {:?}", d),
            ToolEvent::Complete(r) => println!("Complete: {:?}", r),
        }
    }

    Ok(())
}`;
  }

  private generateExampleParams(tool: MCPTool): Record<string, any> {
    const params: Record<string, any> = {};

    if (tool.examples && tool.examples.length > 0) {
      return tool.examples[0].input;
    }

    // Generate example params based on schema
    if (tool.inputSchema.properties) {
      Object.entries(tool.inputSchema.properties).forEach(([name, prop]: [string, any]) => {
        if (prop.examples && prop.examples.length > 0) {
          params[name] = prop.examples[0];
        } else if (prop.enum && prop.enum.length > 0) {
          params[name] = prop.enum[0];
        } else {
          params[name] = this.getDefaultValue(prop);
        }
      });
    }

    return params;
  }

  private getDefaultValue(prop: any): any {
    if (prop.default !== undefined) return prop.default;

    switch (prop.type) {
      case 'string': return 'example';
      case 'number': return 42;
      case 'boolean': return true;
      case 'array': return [];
      case 'object': return {};
      default: return null;
    }
  }

  private findRelatedTools(tool: MCPTool): string[] {
    // This would typically query a database or use more sophisticated matching
    const related: string[] = [];

    // Add tools from the same category
    if (tool.category === 'knowledge-graph') {
      related.push('knowledge-query', 'knowledge-update', 'graph-visualize');
    } else if (tool.category === 'cognitive') {
      related.push('pattern-match', 'learning-update', 'cognitive-analyze');
    } else if (tool.category === 'neural') {
      related.push('neural-scan', 'activity-monitor', 'network-state');
    }

    return related.filter(id => id !== tool.id);
  }

  private exportMarkdown(tools: MCPTool[]): string {
    let markdown = '# LLMKG Tool Documentation\n\n';
    markdown += `Generated on ${new Date().toISOString()}\n\n`;
    markdown += '## Table of Contents\n\n';

    // Group tools by category
    const toolsByCategory = tools.reduce((acc, tool) => {
      if (!acc[tool.category]) acc[tool.category] = [];
      acc[tool.category].push(tool);
      return acc;
    }, {} as Record<string, MCPTool[]>);

    // Generate TOC
    Object.entries(toolsByCategory).forEach(([category, categoryTools]) => {
      markdown += `- [${this.toPascalCase(category)} Tools](#${category}-tools)\n`;
      categoryTools.forEach(tool => {
        markdown += `  - [${tool.name}](#${tool.id})\n`;
      });
    });

    markdown += '\n---\n\n';

    // Generate documentation for each tool
    Object.entries(toolsByCategory).forEach(([category, categoryTools]) => {
      markdown += `## ${this.toPascalCase(category)} Tools\n\n`;

      categoryTools.forEach(tool => {
        const doc = this.generateDocumentation(tool);
        
        markdown += `### ${tool.name} {#${tool.id}}\n\n`;
        markdown += `**Version**: ${tool.version}\n\n`;
        markdown += `${doc.description}\n\n`;

        // Parameters
        if (doc.parameters.length > 0) {
          markdown += '#### Parameters\n\n';
          markdown += '| Name | Type | Required | Description |\n';
          markdown += '|------|------|----------|-------------|\n';
          doc.parameters.forEach(param => {
            markdown += `| ${param.name} | \`${param.type}\` | ${param.required ? 'Yes' : 'No'} | ${param.description} |\n`;
          });
          markdown += '\n';
        }

        // Returns
        markdown += '#### Returns\n\n';
        markdown += `- **Type**: \`${doc.returns.type}\`\n`;
        markdown += `- **Description**: ${doc.returns.description}\n\n`;

        // Examples
        if (doc.examples.length > 0) {
          markdown += '#### Examples\n\n';
          doc.examples.forEach(example => {
            markdown += `<details>\n<summary>${example.description || example.language}</summary>\n\n`;
            markdown += `\`\`\`${example.language}\n${example.code}\n\`\`\`\n\n</details>\n\n`;
          });
        }

        // Related tools
        if (doc.relatedTools && doc.relatedTools.length > 0) {
          markdown += '#### Related Tools\n\n';
          doc.relatedTools.forEach(relatedId => {
            markdown += `- [${relatedId}](#${relatedId})\n`;
          });
          markdown += '\n';
        }

        markdown += '---\n\n';
      });
    });

    return markdown;
  }

  private exportHtml(tools: MCPTool[]): string {
    const markdown = this.exportMarkdown(tools);
    // In a real implementation, you'd use a markdown-to-HTML converter
    // For now, we'll return a basic HTML structure
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMKG Tool Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
        pre { background: #f5f5f5; padding: 16px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f5f5f5; }
    </style>
</head>
<body>
    <div class="container">
        ${markdown} <!-- This would be converted from markdown -->
    </div>
</body>
</html>`;
  }

  private exportPdf(tools: MCPTool[]): string {
    // In a real implementation, you'd use a library like puppeteer or wkhtmltopdf
    // For now, we'll throw an error indicating PDF export needs additional setup
    throw new Error('PDF export requires additional dependencies. Please install @llmkg/pdf-generator');
  }

  private toPascalCase(str: string): string {
    return str
      .split(/[-_]/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join('');
  }

  private toSnakeCase(str: string): string {
    return str
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '')
      .replace(/-/g, '_');
  }
}

// Export singleton instance
export const documentationGenerator = new DocumentationGenerator();