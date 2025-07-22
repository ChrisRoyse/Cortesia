import { JSONSchema, ParameterDoc } from '../types';

/**
 * Utilities for parsing and analyzing JSON schemas
 */
export class SchemaParser {
  /**
   * Parse a JSON schema and extract parameter documentation
   */
  static extractParameters(schema: JSONSchema): ParameterDoc[] {
    const parameters: ParameterDoc[] = [];

    if (schema.properties) {
      Object.entries(schema.properties).forEach(([name, propSchema]) => {
        parameters.push(this.parseParameter(name, propSchema, schema.required || []));
      });
    }

    return parameters;
  }

  /**
   * Parse a single parameter from schema
   */
  private static parseParameter(
    name: string,
    schema: any,
    requiredFields: string[]
  ): ParameterDoc {
    return {
      name,
      type: this.getTypeString(schema),
      description: schema.description || this.generateDescription(name, schema),
      required: requiredFields.includes(name),
      default: schema.default,
      examples: schema.examples || this.generateExamples(schema)
    };
  }

  /**
   * Generate a human-readable type string from schema
   */
  static getTypeString(schema: any): string {
    if (schema.enum) {
      return schema.enum.map((v: any) => `'${v}'`).join(' | ');
    }

    if (schema.type === 'array') {
      const itemType = schema.items ? this.getTypeString(schema.items) : 'any';
      return `${itemType}[]`;
    }

    if (schema.type === 'object') {
      if (schema.properties) {
        const props = Object.keys(schema.properties);
        if (props.length <= 3) {
          return `{ ${props.join(', ')} }`;
        }
        return `object (${props.length} properties)`;
      }
      return 'object';
    }

    if (schema.anyOf) {
      return schema.anyOf.map((s: any) => this.getTypeString(s)).join(' | ');
    }

    if (schema.allOf) {
      return schema.allOf.map((s: any) => this.getTypeString(s)).join(' & ');
    }

    if (schema.oneOf) {
      return schema.oneOf.map((s: any) => this.getTypeString(s)).join(' | ');
    }

    if (schema.type) {
      // Add format information if available
      if (schema.format) {
        return `${schema.type} (${schema.format})`;
      }
      
      // Add constraints
      const constraints: string[] = [];
      if (schema.minimum !== undefined) constraints.push(`>=${schema.minimum}`);
      if (schema.maximum !== undefined) constraints.push(`<=${schema.maximum}`);
      if (schema.minLength !== undefined) constraints.push(`minLength: ${schema.minLength}`);
      if (schema.maxLength !== undefined) constraints.push(`maxLength: ${schema.maxLength}`);
      if (schema.pattern) constraints.push(`pattern: ${schema.pattern}`);
      
      if (constraints.length > 0) {
        return `${schema.type} (${constraints.join(', ')})`;
      }
      
      return schema.type;
    }

    return 'any';
  }

  /**
   * Generate description based on parameter name and schema
   */
  private static generateDescription(name: string, schema: any): string {
    // Common parameter descriptions
    const commonDescriptions: Record<string, string> = {
      id: 'Unique identifier',
      name: 'Name of the item',
      description: 'Detailed description',
      query: 'Search query string',
      filter: 'Filter criteria',
      limit: 'Maximum number of results to return',
      offset: 'Number of results to skip',
      page: 'Page number for pagination',
      sort: 'Sort order for results',
      threshold: 'Confidence or similarity threshold',
      timeout: 'Request timeout in milliseconds',
      format: 'Output format',
      include: 'Fields to include in response',
      exclude: 'Fields to exclude from response',
      depth: 'Depth of traversal or recursion',
      mode: 'Operation mode',
      options: 'Additional options',
      metadata: 'Additional metadata',
      tags: 'Tags for categorization',
      timestamp: 'Unix timestamp',
      date: 'Date in ISO 8601 format',
      enabled: 'Whether the feature is enabled',
      active: 'Whether the item is active',
      debug: 'Enable debug mode',
      verbose: 'Enable verbose output'
    };

    // Check for common descriptions
    if (commonDescriptions[name]) {
      return commonDescriptions[name];
    }

    // Generate based on type and constraints
    let description = `The ${name} parameter`;

    if (schema.type === 'boolean') {
      description = `Whether to enable ${name}`;
    } else if (schema.type === 'array') {
      description = `List of ${name}`;
    } else if (schema.enum) {
      description = `${name} selection from predefined values`;
    }

    // Add constraint information
    if (schema.minimum !== undefined || schema.maximum !== undefined) {
      description += ` (${schema.minimum || 'min'} - ${schema.maximum || 'max'})`;
    }

    return description;
  }

  /**
   * Generate examples based on schema type
   */
  private static generateExamples(schema: any): any[] {
    const examples: any[] = [];

    // If schema already has examples, use them
    if (schema.examples && schema.examples.length > 0) {
      return schema.examples;
    }

    // Generate based on type
    switch (schema.type) {
      case 'string':
        if (schema.enum) {
          examples.push(...schema.enum.slice(0, 3));
        } else if (schema.format === 'date-time') {
          examples.push(new Date().toISOString());
        } else if (schema.format === 'email') {
          examples.push('user@example.com');
        } else if (schema.format === 'uri') {
          examples.push('https://example.com');
        } else if (schema.pattern) {
          examples.push('matches-pattern');
        } else {
          examples.push('example-string', 'another-example');
        }
        break;

      case 'number':
      case 'integer':
        if (schema.minimum !== undefined && schema.maximum !== undefined) {
          examples.push(schema.minimum, Math.floor((schema.minimum + schema.maximum) / 2), schema.maximum);
        } else {
          examples.push(0, 42, 100);
        }
        break;

      case 'boolean':
        examples.push(true, false);
        break;

      case 'array':
        if (schema.items) {
          const itemExamples = this.generateExamples(schema.items);
          examples.push([], itemExamples.slice(0, 2));
        } else {
          examples.push([], ['item1', 'item2']);
        }
        break;

      case 'object':
        const obj: any = {};
        if (schema.properties) {
          Object.entries(schema.properties).forEach(([key, prop]: [string, any]) => {
            const propExamples = this.generateExamples(prop);
            if (propExamples.length > 0) {
              obj[key] = propExamples[0];
            }
          });
        }
        examples.push(obj);
        break;
    }

    return examples;
  }

  /**
   * Validate data against schema
   */
  static validate(data: any, schema: JSONSchema): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Type validation
    if (schema.type && !this.validateType(data, schema.type)) {
      errors.push(`Expected type ${schema.type}, got ${typeof data}`);
    }

    // Required properties
    if (schema.type === 'object' && schema.required) {
      schema.required.forEach(prop => {
        if (!(prop in data)) {
          errors.push(`Missing required property: ${prop}`);
        }
      });
    }

    // Enum validation
    if (schema.enum && !schema.enum.includes(data)) {
      errors.push(`Value must be one of: ${schema.enum.join(', ')}`);
    }

    // String constraints
    if (schema.type === 'string' && typeof data === 'string') {
      if (schema.minLength && data.length < schema.minLength) {
        errors.push(`String length must be at least ${schema.minLength}`);
      }
      if (schema.maxLength && data.length > schema.maxLength) {
        errors.push(`String length must not exceed ${schema.maxLength}`);
      }
      if (schema.pattern && !new RegExp(schema.pattern).test(data)) {
        errors.push(`String must match pattern: ${schema.pattern}`);
      }
    }

    // Number constraints
    if ((schema.type === 'number' || schema.type === 'integer') && typeof data === 'number') {
      if (schema.minimum !== undefined && data < schema.minimum) {
        errors.push(`Value must be at least ${schema.minimum}`);
      }
      if (schema.maximum !== undefined && data > schema.maximum) {
        errors.push(`Value must not exceed ${schema.maximum}`);
      }
      if (schema.type === 'integer' && !Number.isInteger(data)) {
        errors.push('Value must be an integer');
      }
    }

    // Array constraints
    if (schema.type === 'array' && Array.isArray(data)) {
      if (schema.minItems && data.length < schema.minItems) {
        errors.push(`Array must have at least ${schema.minItems} items`);
      }
      if (schema.maxItems && data.length > schema.maxItems) {
        errors.push(`Array must not have more than ${schema.maxItems} items`);
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate type
   */
  private static validateType(data: any, type: string): boolean {
    switch (type) {
      case 'string':
        return typeof data === 'string';
      case 'number':
        return typeof data === 'number' && !isNaN(data);
      case 'integer':
        return typeof data === 'number' && Number.isInteger(data);
      case 'boolean':
        return typeof data === 'boolean';
      case 'array':
        return Array.isArray(data);
      case 'object':
        return data !== null && typeof data === 'object' && !Array.isArray(data);
      case 'null':
        return data === null;
      default:
        return true;
    }
  }

  /**
   * Generate TypeScript interface from schema
   */
  static generateTypeScriptInterface(name: string, schema: JSONSchema): string {
    const lines: string[] = [`export interface ${name} {`];

    if (schema.properties) {
      Object.entries(schema.properties).forEach(([propName, propSchema]: [string, any]) => {
        const required = schema.required?.includes(propName) || false;
        const type = this.getTypeScriptType(propSchema);
        const description = propSchema.description;

        if (description) {
          lines.push(`  /** ${description} */`);
        }
        lines.push(`  ${propName}${required ? '' : '?'}: ${type};`);
      });
    }

    lines.push('}');
    return lines.join('\n');
  }

  /**
   * Get TypeScript type from schema
   */
  private static getTypeScriptType(schema: any): string {
    if (schema.enum) {
      return schema.enum.map((v: any) => `'${v}'`).join(' | ');
    }

    if (schema.type === 'array') {
      const itemType = schema.items ? this.getTypeScriptType(schema.items) : 'any';
      return `${itemType}[]`;
    }

    if (schema.type === 'object') {
      if (schema.properties) {
        const props = Object.entries(schema.properties)
          .map(([key, value]: [string, any]) => {
            const required = schema.required?.includes(key) || false;
            return `${key}${required ? '' : '?'}: ${this.getTypeScriptType(value)}`;
          })
          .join('; ');
        return `{ ${props} }`;
      }
      return 'Record<string, any>';
    }

    const typeMap: Record<string, string> = {
      string: 'string',
      number: 'number',
      integer: 'number',
      boolean: 'boolean',
      null: 'null'
    };

    return typeMap[schema.type] || 'any';
  }
}