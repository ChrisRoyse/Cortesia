import React, { useState, useCallback, useMemo, useEffect } from 'react';
import {
  Box,
  TextField,
  Select,
  MenuItem,
  FormControl,
  FormLabel,
  FormControlLabel,
  Checkbox,
  Switch,
  Button,
  IconButton,
  Paper,
  Typography,
  Stack,
  Chip,
  Tooltip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  InputAdornment,
  FormHelperText,
} from '@mui/material';
import {
  Add,
  Remove,
  ExpandMore,
  Info,
  ContentPaste,
  Clear,
  Code,
  Lightbulb,
} from '@mui/icons-material';
import { JSONSchema7 } from 'json-schema';

interface DynamicFormProps {
  schema: JSONSchema7;
  onSubmit: (values: any) => void;
  onChange?: (values: any) => void;
  initialValues?: any;
  examples?: Array<{ name: string; description?: string; input: any }>;
  disabled?: boolean;
}

interface FieldProps {
  schema: JSONSchema7;
  path: string[];
  value: any;
  onChange: (path: string[], value: any) => void;
  required?: boolean;
  disabled?: boolean;
}

const Field: React.FC<FieldProps> = ({
  schema,
  path,
  value,
  onChange,
  required,
  disabled,
}) => {
  const fieldName = path[path.length - 1] || 'root';
  const description = schema.description;

  // Handle different schema types
  if (schema.enum) {
    return (
      <FormControl fullWidth>
        <FormLabel>{fieldName}</FormLabel>
        <Select
          value={value || ''}
          onChange={(e) => onChange(path, e.target.value)}
          disabled={disabled}
          size="small"
        >
          {schema.enum.map((option) => (
            <MenuItem key={String(option)} value={option}>
              {String(option)}
            </MenuItem>
          ))}
        </Select>
        {description && (
          <FormHelperText>{description}</FormHelperText>
        )}
      </FormControl>
    );
  }

  switch (schema.type) {
    case 'string':
      if (schema.format === 'uri' || schema.format === 'url') {
        return (
          <TextField
            fullWidth
            label={fieldName}
            value={value || ''}
            onChange={(e) => onChange(path, e.target.value)}
            required={required}
            disabled={disabled}
            helperText={description}
            type="url"
            size="small"
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Code />
                </InputAdornment>
              ),
            }}
          />
        );
      }

      if (schema.pattern) {
        return (
          <TextField
            fullWidth
            label={fieldName}
            value={value || ''}
            onChange={(e) => onChange(path, e.target.value)}
            required={required}
            disabled={disabled}
            helperText={description || `Pattern: ${schema.pattern}`}
            size="small"
            error={value && !new RegExp(schema.pattern).test(value)}
          />
        );
      }

      return (
        <TextField
          fullWidth
          label={fieldName}
          value={value || ''}
          onChange={(e) => onChange(path, e.target.value)}
          required={required}
          disabled={disabled}
          helperText={description}
          multiline={schema.maxLength && schema.maxLength > 100}
          rows={schema.maxLength && schema.maxLength > 100 ? 4 : 1}
          size="small"
        />
      );

    case 'number':
    case 'integer':
      return (
        <TextField
          fullWidth
          label={fieldName}
          type="number"
          value={value || ''}
          onChange={(e) => onChange(path, e.target.value ? Number(e.target.value) : '')}
          required={required}
          disabled={disabled}
          helperText={description}
          size="small"
          inputProps={{
            min: schema.minimum,
            max: schema.maximum,
            step: schema.type === 'integer' ? 1 : 0.01,
          }}
        />
      );

    case 'boolean':
      return (
        <FormControl fullWidth>
          <FormControlLabel
            control={
              <Switch
                checked={value || false}
                onChange={(e) => onChange(path, e.target.checked)}
                disabled={disabled}
              />
            }
            label={fieldName}
          />
          {description && (
            <FormHelperText>{description}</FormHelperText>
          )}
        </FormControl>
      );

    case 'array':
      return (
        <ArrayField
          schema={schema}
          path={path}
          value={value}
          onChange={onChange}
          disabled={disabled}
        />
      );

    case 'object':
      return (
        <ObjectField
          schema={schema}
          path={path}
          value={value}
          onChange={onChange}
          disabled={disabled}
        />
      );

    default:
      return (
        <Alert severity="warning">
          Unsupported field type: {schema.type}
        </Alert>
      );
  }
};

const ArrayField: React.FC<FieldProps> = ({
  schema,
  path,
  value,
  onChange,
  disabled,
}) => {
  const items = value || [];
  const itemSchema = schema.items as JSONSchema7;
  const fieldName = path[path.length - 1] || 'items';

  const handleAdd = () => {
    const newValue = [...items, getDefaultValue(itemSchema)];
    onChange(path, newValue);
  };

  const handleRemove = (index: number) => {
    const newValue = items.filter((_: any, i: number) => i !== index);
    onChange(path, newValue);
  };

  const handleItemChange = (index: number, itemValue: any) => {
    const newValue = [...items];
    newValue[index] = itemValue;
    onChange(path, newValue);
  };

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Stack spacing={2}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="subtitle2">{fieldName}</Typography>
          <Button
            size="small"
            startIcon={<Add />}
            onClick={handleAdd}
            disabled={disabled || (schema.maxItems && items.length >= schema.maxItems)}
          >
            Add Item
          </Button>
        </Stack>
        
        {schema.description && (
          <Typography variant="caption" color="text.secondary">
            {schema.description}
          </Typography>
        )}

        {items.map((item: any, index: number) => (
          <Stack key={index} direction="row" spacing={1} alignItems="flex-start">
            <Box sx={{ flex: 1 }}>
              <Field
                schema={itemSchema}
                path={[...path, String(index)]}
                value={item}
                onChange={(p, v) => handleItemChange(index, v)}
                disabled={disabled}
              />
            </Box>
            <IconButton
              size="small"
              onClick={() => handleRemove(index)}
              disabled={disabled}
            >
              <Remove />
            </IconButton>
          </Stack>
        ))}

        {items.length === 0 && (
          <Alert severity="info">
            No items yet. Click "Add Item" to create one.
          </Alert>
        )}
      </Stack>
    </Paper>
  );
};

const ObjectField: React.FC<FieldProps> = ({
  schema,
  path,
  value,
  onChange,
  disabled,
}) => {
  const properties = schema.properties || {};
  const required = schema.required || [];
  const fieldName = path[path.length - 1];

  const handleFieldChange = (fieldPath: string[], fieldValue: any) => {
    const newValue = { ...(value || {}) };
    const fieldKey = fieldPath[fieldPath.length - 1];
    newValue[fieldKey] = fieldValue;
    onChange(path, newValue);
  };

  const isExpanded = path.length === 0; // Root object is expanded by default

  return (
    <Accordion defaultExpanded={isExpanded} variant="outlined">
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Typography variant="subtitle2">
          {fieldName || 'Parameters'}
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={2}>
          {Object.entries(properties).map(([key, propSchema]) => (
            <Field
              key={key}
              schema={propSchema as JSONSchema7}
              path={[...path, key]}
              value={value?.[key]}
              onChange={handleFieldChange}
              required={required.includes(key)}
              disabled={disabled}
            />
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
};

const getDefaultValue = (schema: JSONSchema7): any => {
  if (schema.default !== undefined) return schema.default;
  
  switch (schema.type) {
    case 'string': return '';
    case 'number': return schema.minimum || 0;
    case 'integer': return schema.minimum || 0;
    case 'boolean': return false;
    case 'array': return [];
    case 'object': return {};
    default: return null;
  }
};

const DynamicForm: React.FC<DynamicFormProps> = ({
  schema,
  onSubmit,
  onChange,
  initialValues,
  examples,
  disabled,
}) => {
  const [values, setValues] = useState(initialValues || getDefaultValue(schema));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showExamples, setShowExamples] = useState(false);

  useEffect(() => {
    if (onChange) {
      onChange(values);
    }
  }, [values, onChange]);

  const handleFieldChange = useCallback((path: string[], value: any) => {
    if (path.length === 0) {
      setValues(value);
    } else {
      const newValues = { ...values };
      let current = newValues;
      
      for (let i = 0; i < path.length - 1; i++) {
        if (!current[path[i]]) {
          current[path[i]] = {};
        }
        current = current[path[i]];
      }
      
      current[path[path.length - 1]] = value;
      setValues(newValues);
    }
  }, [values]);

  const handleLoadExample = useCallback((example: any) => {
    setValues(example.input);
    setShowExamples(false);
  }, []);

  const handlePasteJson = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      const parsed = JSON.parse(text);
      setValues(parsed);
    } catch (error) {
      setErrors({ paste: 'Invalid JSON in clipboard' });
    }
  }, []);

  const handleClear = useCallback(() => {
    setValues(getDefaultValue(schema));
  }, [schema]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(values);
  }, [values, onSubmit]);

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Stack spacing={2}>
        {/* Toolbar */}
        <Stack direction="row" spacing={1} justifyContent="flex-end">
          {examples && examples.length > 0 && (
            <Button
              size="small"
              startIcon={<Lightbulb />}
              onClick={() => setShowExamples(!showExamples)}
            >
              Examples
            </Button>
          )}
          <Button
            size="small"
            startIcon={<ContentPaste />}
            onClick={handlePasteJson}
          >
            Paste JSON
          </Button>
          <Button
            size="small"
            startIcon={<Clear />}
            onClick={handleClear}
          >
            Clear
          </Button>
        </Stack>

        {/* Examples */}
        {showExamples && examples && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Examples
            </Typography>
            <Stack spacing={1}>
              {examples.map((example, index) => (
                <Stack
                  key={index}
                  direction="row"
                  spacing={1}
                  alignItems="center"
                  sx={{
                    p: 1,
                    borderRadius: 1,
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                  onClick={() => handleLoadExample(example)}
                >
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="body2">{example.name}</Typography>
                    {example.description && (
                      <Typography variant="caption" color="text.secondary">
                        {example.description}
                      </Typography>
                    )}
                  </Box>
                  <Chip label="Load" size="small" color="primary" />
                </Stack>
              ))}
            </Stack>
          </Paper>
        )}

        {/* Form Fields */}
        <Field
          schema={schema}
          path={[]}
          value={values}
          onChange={handleFieldChange}
          disabled={disabled}
        />

        {/* Error Display */}
        {Object.keys(errors).length > 0 && (
          <Alert severity="error" onClose={() => setErrors({})}>
            {Object.values(errors).join(', ')}
          </Alert>
        )}
      </Stack>
    </Box>
  );
};

export default DynamicForm;