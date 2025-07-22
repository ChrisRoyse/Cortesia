import React, { useState, useMemo, useCallback } from 'react';
import {
  Box,
  IconButton,
  Typography,
  TextField,
  InputAdornment,
  Collapse,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  ExpandMore,
  ChevronRight,
  ContentCopy,
  Search,
  Clear,
} from '@mui/icons-material';
import { copyToClipboard } from '../../utils/dataFormatters';

interface HighlightRule {
  path?: string[];
  key?: string;
  value?: any;
  color: string;
}

interface JsonViewerProps {
  data: any;
  theme?: 'light' | 'dark';
  expandLevel?: number;
  onNodeClick?: (path: string[], value: any) => void;
  highlighting?: HighlightRule[];
  searchable?: boolean;
}

interface JsonNodeProps {
  keyName: string;
  value: any;
  path: string[];
  expandLevel: number;
  currentLevel: number;
  onNodeClick?: (path: string[], value: any) => void;
  highlighting?: HighlightRule[];
  searchTerm?: string;
  theme: 'light' | 'dark';
}

const JsonNode: React.FC<JsonNodeProps> = ({
  keyName,
  value,
  path,
  expandLevel,
  currentLevel,
  onNodeClick,
  highlighting,
  searchTerm,
  theme,
}) => {
  const muiTheme = useTheme();
  const [isExpanded, setIsExpanded] = useState(currentLevel < expandLevel);

  const isObject = value !== null && typeof value === 'object';
  const isArray = Array.isArray(value);

  const handleToggle = () => {
    setIsExpanded(!isExpanded);
  };

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyToClipboard(value);
  };

  const getHighlightColor = useCallback(() => {
    if (!highlighting) return null;

    for (const rule of highlighting) {
      if (rule.path && JSON.stringify(rule.path) === JSON.stringify(path)) {
        return rule.color;
      }
      if (rule.key && keyName === rule.key) {
        return rule.color;
      }
      if (rule.value !== undefined && value === rule.value) {
        return rule.color;
      }
    }
    return null;
  }, [highlighting, path, keyName, value]);

  const isHighlighted = useCallback(() => {
    if (!searchTerm) return false;
    const searchLower = searchTerm.toLowerCase();
    
    if (keyName.toLowerCase().includes(searchLower)) return true;
    
    if (!isObject) {
      return String(value).toLowerCase().includes(searchLower);
    }
    
    return false;
  }, [searchTerm, keyName, value, isObject]);

  const syntaxColors = {
    light: {
      key: '#0969da',
      string: '#0a3069',
      number: '#cf222e',
      boolean: '#8250df',
      null: '#6e7781',
      bracket: '#24292f',
    },
    dark: {
      key: '#79c0ff',
      string: '#a5d6ff',
      number: '#ff7b72',
      boolean: '#d2a8ff',
      null: '#8b949e',
      bracket: '#c9d1d9',
    },
  };

  const colors = syntaxColors[theme];
  const highlightColor = getHighlightColor();
  const isSearchHighlight = isHighlighted();

  const renderValue = () => {
    if (value === null) {
      return <span style={{ color: colors.null }}>null</span>;
    }

    if (typeof value === 'boolean') {
      return <span style={{ color: colors.boolean }}>{String(value)}</span>;
    }

    if (typeof value === 'number') {
      return <span style={{ color: colors.number }}>{value}</span>;
    }

    if (typeof value === 'string') {
      return <span style={{ color: colors.string }}>"{value}"</span>;
    }

    if (isArray) {
      return (
        <span style={{ color: colors.bracket }}>
          [{!isExpanded && `...${value.length} items`}]
        </span>
      );
    }

    if (isObject) {
      const keys = Object.keys(value);
      return (
        <span style={{ color: colors.bracket }}>
          {'{'}
          {!isExpanded && `...${keys.length} keys`}
          {'}'}
        </span>
      );
    }

    return String(value);
  };

  const nodeStyle = {
    backgroundColor: highlightColor
      ? alpha(highlightColor, 0.1)
      : isSearchHighlight
      ? alpha(muiTheme.palette.warning.main, 0.1)
      : 'transparent',
    borderRadius: 1,
    transition: 'background-color 0.3s',
  };

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          py: 0.5,
          px: 1,
          ...nodeStyle,
          '&:hover': {
            backgroundColor: alpha(muiTheme.palette.primary.main, 0.05),
          },
        }}
        onClick={() => onNodeClick && onNodeClick(path, value)}
      >
        {isObject && (
          <IconButton
            size="small"
            onClick={handleToggle}
            sx={{ p: 0, mr: 0.5 }}
          >
            {isExpanded ? <ExpandMore /> : <ChevronRight />}
          </IconButton>
        )}

        <Box sx={{ flex: 1, display: 'flex', alignItems: 'flex-start' }}>
          <Typography
            component="span"
            sx={{
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              color: colors.key,
              mr: 1,
            }}
          >
            {keyName}:
          </Typography>

          <Box sx={{ flex: 1 }}>
            <Typography
              component="span"
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                wordBreak: 'break-word',
              }}
            >
              {renderValue()}
            </Typography>
          </Box>
        </Box>

        <Tooltip title="Copy value">
          <IconButton
            size="small"
            onClick={handleCopy}
            sx={{
              p: 0.5,
              ml: 1,
              opacity: 0,
              transition: 'opacity 0.2s',
              '.MuiBox-root:hover &': {
                opacity: 1,
              },
            }}
          >
            <ContentCopy fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {isObject && (
        <Collapse in={isExpanded}>
          <Box sx={{ pl: 3 }}>
            {isArray
              ? value.map((item: any, index: number) => (
                  <JsonNode
                    key={index}
                    keyName={String(index)}
                    value={item}
                    path={[...path, String(index)]}
                    expandLevel={expandLevel}
                    currentLevel={currentLevel + 1}
                    onNodeClick={onNodeClick}
                    highlighting={highlighting}
                    searchTerm={searchTerm}
                    theme={theme}
                  />
                ))
              : Object.entries(value).map(([key, val]) => (
                  <JsonNode
                    key={key}
                    keyName={key}
                    value={val}
                    path={[...path, key]}
                    expandLevel={expandLevel}
                    currentLevel={currentLevel + 1}
                    onNodeClick={onNodeClick}
                    highlighting={highlighting}
                    searchTerm={searchTerm}
                    theme={theme}
                  />
                ))}
          </Box>
        </Collapse>
      )}
    </Box>
  );
};

const JsonViewer: React.FC<JsonViewerProps> = ({
  data,
  theme = 'light',
  expandLevel = 2,
  onNodeClick,
  highlighting,
  searchable = true,
}) => {
  const muiTheme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');

  const handlePathClick = useCallback(
    (path: string[], value: any) => {
      if (onNodeClick) {
        onNodeClick(path, value);
      }
    },
    [onNodeClick]
  );

  const effectiveTheme = theme || muiTheme.palette.mode;

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {searchable && (
        <Box sx={{ mb: 2 }}>
          <TextField
            size="small"
            fullWidth
            placeholder="Search keys and values..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
              endAdornment: searchTerm && (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setSearchTerm('')}
                  >
                    <Clear />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Box>
      )}

      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          backgroundColor: effectiveTheme === 'dark' 
            ? alpha(muiTheme.palette.background.paper, 0.4)
            : alpha(muiTheme.palette.background.paper, 0.8),
          borderRadius: 1,
          border: 1,
          borderColor: 'divider',
          p: 2,
        }}
      >
        {data === undefined ? (
          <Typography color="text.secondary" sx={{ fontFamily: 'monospace' }}>
            undefined
          </Typography>
        ) : data === null ? (
          <Typography sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
            null
          </Typography>
        ) : typeof data === 'object' ? (
          <JsonNode
            keyName="root"
            value={data}
            path={[]}
            expandLevel={expandLevel}
            currentLevel={0}
            onNodeClick={handlePathClick}
            highlighting={highlighting}
            searchTerm={searchTerm}
            theme={effectiveTheme}
          />
        ) : (
          <Typography sx={{ fontFamily: 'monospace' }}>
            {JSON.stringify(data)}
          </Typography>
        )}
      </Box>

      {onNodeClick && (
        <Box sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            Click any node to view its path
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default JsonViewer;