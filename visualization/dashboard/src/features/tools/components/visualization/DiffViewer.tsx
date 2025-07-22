import React, { useMemo, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  TextField,
  InputAdornment,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Add,
  Remove,
  SwapHoriz,
  Search,
} from '@mui/icons-material';
import { diffLines, diffJson, Change } from 'diff';

interface DiffViewerProps {
  before: any;
  after: any;
  beforeLabel?: string;
  afterLabel?: string;
  viewMode?: 'unified' | 'split';
  ignoreWhitespace?: boolean;
}

interface ProcessedChange extends Change {
  lineNumbers: {
    before: number | null;
    after: number | null;
  };
  type: 'add' | 'remove' | 'unchanged';
}

const DiffViewer: React.FC<DiffViewerProps> = ({
  before,
  after,
  beforeLabel = 'Before',
  afterLabel = 'After',
  viewMode = 'unified',
  ignoreWhitespace = false,
}) => {
  const theme = useTheme();
  const [currentViewMode, setCurrentViewMode] = useState(viewMode);
  const [searchTerm, setSearchTerm] = useState('');

  const { changes, stats } = useMemo(() => {
    // Convert to string if needed
    const beforeStr = typeof before === 'string' 
      ? before 
      : JSON.stringify(before, null, 2);
    const afterStr = typeof after === 'string' 
      ? after 
      : JSON.stringify(after, null, 2);

    // Calculate diff
    const rawChanges = diffLines(beforeStr, afterStr, { ignoreWhitespace });

    // Process changes with line numbers
    let beforeLineNum = 1;
    let afterLineNum = 1;
    const processedChanges: ProcessedChange[] = [];
    let additions = 0;
    let deletions = 0;
    let unchanged = 0;

    rawChanges.forEach((change) => {
      const lines = change.value.split('\n').filter(line => line !== '');
      
      lines.forEach((line) => {
        const processedChange: ProcessedChange = {
          ...change,
          value: line,
          lineNumbers: {
            before: null,
            after: null,
          },
          type: 'unchanged',
        };

        if (change.added) {
          processedChange.lineNumbers.after = afterLineNum++;
          processedChange.type = 'add';
          additions++;
        } else if (change.removed) {
          processedChange.lineNumbers.before = beforeLineNum++;
          processedChange.type = 'remove';
          deletions++;
        } else {
          processedChange.lineNumbers.before = beforeLineNum++;
          processedChange.lineNumbers.after = afterLineNum++;
          processedChange.type = 'unchanged';
          unchanged++;
        }

        processedChanges.push(processedChange);
      });
    });

    return {
      changes: processedChanges,
      stats: { additions, deletions, unchanged },
    };
  }, [before, after, ignoreWhitespace]);

  const filteredChanges = useMemo(() => {
    if (!searchTerm) return changes;
    
    const searchLower = searchTerm.toLowerCase();
    return changes.filter(change => 
      change.value.toLowerCase().includes(searchLower)
    );
  }, [changes, searchTerm]);

  const getChangeColor = (type: 'add' | 'remove' | 'unchanged') => {
    switch (type) {
      case 'add':
        return {
          bg: alpha(theme.palette.success.main, 0.1),
          border: theme.palette.success.main,
          icon: <Add fontSize="small" />,
        };
      case 'remove':
        return {
          bg: alpha(theme.palette.error.main, 0.1),
          border: theme.palette.error.main,
          icon: <Remove fontSize="small" />,
        };
      default:
        return {
          bg: 'transparent',
          border: 'transparent',
          icon: null,
        };
    }
  };

  const renderUnifiedView = () => (
    <Box sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
      {filteredChanges.map((change, index) => {
        const style = getChangeColor(change.type);
        const lineNum = change.lineNumbers.after || change.lineNumbers.before;

        return (
          <Box
            key={index}
            sx={{
              display: 'flex',
              backgroundColor: style.bg,
              borderLeft: `3px solid ${style.border}`,
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.05),
              },
            }}
          >
            <Box
              sx={{
                width: 60,
                px: 1,
                py: 0.5,
                borderRight: 1,
                borderColor: 'divider',
                color: 'text.secondary',
                textAlign: 'right',
                userSelect: 'none',
              }}
            >
              {lineNum}
            </Box>
            <Box
              sx={{
                width: 30,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRight: 1,
                borderColor: 'divider',
              }}
            >
              {style.icon}
            </Box>
            <Box
              sx={{
                flex: 1,
                px: 2,
                py: 0.5,
                whiteSpace: 'pre',
                overflow: 'auto',
              }}
            >
              {change.value}
            </Box>
          </Box>
        );
      })}
    </Box>
  );

  const renderSplitView = () => {
    const beforeChanges = changes.filter(c => c.type === 'remove' || c.type === 'unchanged');
    const afterChanges = changes.filter(c => c.type === 'add' || c.type === 'unchanged');

    const maxLength = Math.max(beforeChanges.length, afterChanges.length);
    const rows = [];

    for (let i = 0; i < maxLength; i++) {
      const beforeChange = beforeChanges[i];
      const afterChange = afterChanges[i];
      rows.push({ before: beforeChange, after: afterChange });
    }

    return (
      <Box sx={{ display: 'flex', gap: 2, fontFamily: 'monospace', fontSize: '0.875rem' }}>
        {/* Before column */}
        <Box sx={{ flex: 1 }}>
          <Typography
            variant="subtitle2"
            sx={{
              p: 1,
              bgcolor: 'background.default',
              borderBottom: 1,
              borderColor: 'divider',
              position: 'sticky',
              top: 0,
              zIndex: 1,
            }}
          >
            {beforeLabel}
          </Typography>
          <Box>
            {rows.map((row, index) => {
              const change = row.before;
              if (!change) {
                return <Box key={index} sx={{ height: 28 }} />;
              }

              const style = getChangeColor(change.type);
              return (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    backgroundColor: style.bg,
                    borderLeft: `3px solid ${style.border}`,
                  }}
                >
                  <Box
                    sx={{
                      width: 50,
                      px: 1,
                      py: 0.5,
                      borderRight: 1,
                      borderColor: 'divider',
                      color: 'text.secondary',
                      textAlign: 'right',
                    }}
                  >
                    {change.lineNumbers.before}
                  </Box>
                  <Box
                    sx={{
                      flex: 1,
                      px: 2,
                      py: 0.5,
                      whiteSpace: 'pre',
                      overflow: 'auto',
                    }}
                  >
                    {change.value}
                  </Box>
                </Box>
              );
            })}
          </Box>
        </Box>

        {/* After column */}
        <Box sx={{ flex: 1 }}>
          <Typography
            variant="subtitle2"
            sx={{
              p: 1,
              bgcolor: 'background.default',
              borderBottom: 1,
              borderColor: 'divider',
              position: 'sticky',
              top: 0,
              zIndex: 1,
            }}
          >
            {afterLabel}
          </Typography>
          <Box>
            {rows.map((row, index) => {
              const change = row.after;
              if (!change) {
                return <Box key={index} sx={{ height: 28 }} />;
              }

              const style = getChangeColor(change.type);
              return (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    backgroundColor: style.bg,
                    borderLeft: `3px solid ${style.border}`,
                  }}
                >
                  <Box
                    sx={{
                      width: 50,
                      px: 1,
                      py: 0.5,
                      borderRight: 1,
                      borderColor: 'divider',
                      color: 'text.secondary',
                      textAlign: 'right',
                    }}
                  >
                    {change.lineNumbers.after}
                  </Box>
                  <Box
                    sx={{
                      flex: 1,
                      px: 2,
                      py: 0.5,
                      whiteSpace: 'pre',
                      overflow: 'auto',
                    }}
                  >
                    {change.value}
                  </Box>
                </Box>
              );
            })}
          </Box>
        </Box>
      </Box>
    );
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6">Diff View</Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`+${stats.additions}`}
              size="small"
              color="success"
              variant="outlined"
            />
            <Chip
              label={`-${stats.deletions}`}
              size="small"
              color="error"
              variant="outlined"
            />
            <Chip
              label={`${stats.unchanged} unchanged`}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <TextField
            size="small"
            placeholder="Search in diff..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
            sx={{ width: 200 }}
          />

          <ToggleButtonGroup
            value={currentViewMode}
            exclusive
            onChange={(e, value) => value && setCurrentViewMode(value)}
            size="small"
          >
            <ToggleButton value="unified">
              Unified
            </ToggleButton>
            <ToggleButton value="split">
              <SwapHoriz sx={{ mr: 0.5 }} />
              Split
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Content */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          bgcolor: theme.palette.mode === 'dark'
            ? alpha(theme.palette.background.paper, 0.4)
            : alpha(theme.palette.grey[50], 0.5),
        }}
      >
        {currentViewMode === 'unified' ? renderUnifiedView() : renderSplitView()}
      </Box>

      {/* Legend */}
      <Box
        sx={{
          p: 1,
          borderTop: 1,
          borderColor: 'divider',
          display: 'flex',
          gap: 2,
          bgcolor: 'background.default',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 16,
              height: 16,
              bgcolor: alpha(theme.palette.success.main, 0.2),
              border: 1,
              borderColor: theme.palette.success.main,
            }}
          />
          <Typography variant="caption">Added</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 16,
              height: 16,
              bgcolor: alpha(theme.palette.error.main, 0.2),
              border: 1,
              borderColor: theme.palette.error.main,
            }}
          />
          <Typography variant="caption">Removed</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 16,
              height: 16,
              bgcolor: 'transparent',
              border: 1,
              borderColor: 'divider',
            }}
          />
          <Typography variant="caption">Unchanged</Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default DiffViewer;