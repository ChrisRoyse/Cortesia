import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { FixedSizeList as List, VariableSizeList } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { useAppSelector } from '../../stores';

export interface Column<T = any> {
  key: string;
  title: string;
  width?: number;
  minWidth?: number;
  maxWidth?: number;
  resizable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, record: T, index: number) => React.ReactNode;
  sorter?: (a: T, b: T) => number;
  filter?: {
    type: 'text' | 'select' | 'date' | 'number' | 'boolean';
    options?: Array<{ label: string; value: any }>;
  };
  align?: 'left' | 'center' | 'right';
  fixed?: 'left' | 'right';
  className?: string;
}

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

export interface FilterConfig {
  [key: string]: any;
}

export interface DataGridProps<T = any> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  rowHeight?: number;
  headerHeight?: number;
  width?: number;
  height?: number;
  pageSize?: number;
  pagination?: boolean;
  virtualScrolling?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  selectable?: boolean;
  multiSelect?: boolean;
  selectedRowKeys?: (string | number)[];
  rowKey?: string | ((record: T) => string | number);
  onRowSelect?: (selectedRows: T[], selectedRowKeys: (string | number)[]) => void;
  onRowClick?: (record: T, index: number) => void;
  onSort?: (sortConfig: SortConfig | null) => void;
  onFilter?: (filters: FilterConfig) => void;
  emptyText?: string;
  className?: string;
  rowClassName?: string | ((record: T, index: number) => string);
  style?: React.CSSProperties;
}

interface RowProps<T> {
  index: number;
  style: React.CSSProperties;
  data: {
    items: T[];
    columns: Column<T>[];
    selectedRowKeys: Set<string | number>;
    onRowSelect: (key: string | number, selected: boolean) => void;
    onRowClick?: (record: T, index: number) => void;
    rowKey: (record: T) => string | number;
    theme: string;
    selectable: boolean;
    rowClassName?: string | ((record: T, index: number) => string);
  };
}

// Virtual row component
const VirtualRow = <T,>({ index, style, data }: RowProps<T>) => {
  const {
    items,
    columns,
    selectedRowKeys,
    onRowSelect,
    onRowClick,
    rowKey,
    theme,
    selectable,
    rowClassName,
  } = data;

  const record = items[index];
  const key = rowKey(record);
  const isSelected = selectedRowKeys.has(key);

  const handleRowClick = useCallback(() => {
    onRowClick?.(record, index);
  }, [record, index, onRowClick]);

  const handleSelectChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    onRowSelect(key, e.target.checked);
  }, [key, onRowSelect]);

  const getRowClassName = () => {
    let className = 'data-grid-row';
    
    if (typeof rowClassName === 'function') {
      className += ` ${rowClassName(record, index)}`;
    } else if (rowClassName) {
      className += ` ${rowClassName}`;
    }
    
    if (isSelected) {
      className += ' selected';
    }
    
    if (index % 2 === 0) {
      className += ' even';
    } else {
      className += ' odd';
    }
    
    return className;
  };

  return (
    <div
      style={{
        ...style,
        display: 'flex',
        alignItems: 'center',
        borderBottom: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        backgroundColor: isSelected 
          ? (theme === 'dark' ? '#1e40af20' : '#3b82f620')
          : (index % 2 === 0 
            ? (theme === 'dark' ? '#1e293b' : '#ffffff') 
            : (theme === 'dark' ? '#0f172a' : '#f9fafb')),
        cursor: onRowClick ? 'pointer' : 'default',
      }}
      className={getRowClassName()}
      onClick={handleRowClick}
    >
      {selectable && (
        <div style={{ 
          width: '40px', 
          paddingLeft: '12px',
          display: 'flex',
          justifyContent: 'center' 
        }}>
          <input
            type="checkbox"
            checked={isSelected}
            onChange={handleSelectChange}
            style={{ 
              width: '16px', 
              height: '16px',
              accentColor: theme === 'dark' ? '#3b82f6' : '#2563eb'
            }}
          />
        </div>
      )}
      
      {columns.map((column) => {
        const value = record[column.key as keyof T];
        const cellContent = column.render ? column.render(value, record, index) : String(value ?? '');
        
        return (
          <div
            key={column.key}
            style={{
              width: column.width || 'auto',
              minWidth: column.minWidth || 80,
              maxWidth: column.maxWidth,
              padding: '8px 12px',
              textAlign: column.align || 'left',
              color: theme === 'dark' ? '#ffffff' : '#374151',
              fontSize: '14px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {cellContent}
          </div>
        );
      })}
    </div>
  );
};

// Header component
const DataGridHeader = <T,>({
  columns,
  sortConfig,
  onSort,
  onFilter,
  filters,
  theme,
  selectable,
  selectedRowKeys,
  totalRows,
  onSelectAll,
}: {
  columns: Column<T>[];
  sortConfig: SortConfig | null;
  onSort: (key: string) => void;
  onFilter: (key: string, value: any) => void;
  filters: FilterConfig;
  theme: string;
  selectable: boolean;
  selectedRowKeys: Set<string | number>;
  totalRows: number;
  onSelectAll: (selected: boolean) => void;
}) => {
  const allSelected = selectedRowKeys.size === totalRows && totalRows > 0;
  const indeterminate = selectedRowKeys.size > 0 && selectedRowKeys.size < totalRows;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        borderBottom: `2px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        backgroundColor: theme === 'dark' ? '#0f172a' : '#f9fafb',
        fontWeight: '600',
        fontSize: '14px',
        color: theme === 'dark' ? '#ffffff' : '#374151',
      }}
    >
      {selectable && (
        <div style={{ 
          width: '40px', 
          paddingLeft: '12px',
          display: 'flex',
          justifyContent: 'center' 
        }}>
          <input
            type="checkbox"
            checked={allSelected}
            ref={(input) => {
              if (input) input.indeterminate = indeterminate;
            }}
            onChange={(e) => onSelectAll(e.target.checked)}
            style={{ 
              width: '16px', 
              height: '16px',
              accentColor: theme === 'dark' ? '#3b82f6' : '#2563eb'
            }}
          />
        </div>
      )}
      
      {columns.map((column) => (
        <div
          key={column.key}
          style={{
            width: column.width || 'auto',
            minWidth: column.minWidth || 80,
            maxWidth: column.maxWidth,
            padding: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              cursor: column.sortable ? 'pointer' : 'default',
            }}
            onClick={() => column.sortable && onSort(column.key)}
          >
            <span>{column.title}</span>
            {column.sortable && (
              <span style={{ fontSize: '10px', color: theme === 'dark' ? '#9ca3af' : '#6b7280' }}>
                {sortConfig?.key === column.key
                  ? sortConfig.direction === 'asc' ? '▲' : '▼'
                  : '⇅'}
              </span>
            )}
          </div>
          
          {column.filterable && (
            <div style={{ marginTop: '4px' }}>
              {column.filter?.type === 'select' ? (
                <select
                  value={filters[column.key] || ''}
                  onChange={(e) => onFilter(column.key, e.target.value)}
                  style={{
                    width: '100%',
                    padding: '2px 4px',
                    border: `1px solid ${theme === 'dark' ? '#374151' : '#d1d5db'}`,
                    borderRadius: '4px',
                    backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
                    color: theme === 'dark' ? '#ffffff' : '#374151',
                    fontSize: '12px',
                  }}
                >
                  <option value="">All</option>
                  {column.filter.options?.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={column.filter?.type || 'text'}
                  value={filters[column.key] || ''}
                  onChange={(e) => onFilter(column.key, e.target.value)}
                  placeholder={`Filter ${column.title.toLowerCase()}`}
                  style={{
                    width: '100%',
                    padding: '2px 4px',
                    border: `1px solid ${theme === 'dark' ? '#374151' : '#d1d5db'}`,
                    borderRadius: '4px',
                    backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
                    color: theme === 'dark' ? '#ffffff' : '#374151',
                    fontSize: '12px',
                  }}
                />
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// Pagination component
const DataGridPagination = ({
  currentPage,
  pageSize,
  total,
  onPageChange,
  onPageSizeChange,
  theme,
}: {
  currentPage: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  theme: string;
}) => {
  const totalPages = Math.ceil(total / pageSize);
  const startIndex = (currentPage - 1) * pageSize + 1;
  const endIndex = Math.min(currentPage * pageSize, total);

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px',
        borderTop: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        backgroundColor: theme === 'dark' ? '#0f172a' : '#f9fafb',
        fontSize: '14px',
        color: theme === 'dark' ? '#ffffff' : '#374151',
      }}
    >
      <div>
        Showing {startIndex} to {endIndex} of {total} entries
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>Page size:</span>
        <select
          value={pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
          style={{
            padding: '4px 8px',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#d1d5db'}`,
            borderRadius: '4px',
            backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
            color: theme === 'dark' ? '#ffffff' : '#374151',
          }}
        >
          {[10, 25, 50, 100].map(size => (
            <option key={size} value={size}>{size}</option>
          ))}
        </select>
        
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          style={{
            padding: '4px 8px',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#d1d5db'}`,
            borderRadius: '4px',
            backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
            color: theme === 'dark' ? '#ffffff' : '#374151',
            cursor: currentPage === 1 ? 'not-allowed' : 'pointer',
            opacity: currentPage === 1 ? 0.5 : 1,
          }}
        >
          Previous
        </button>
        
        <span>
          Page {currentPage} of {totalPages}
        </span>
        
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          style={{
            padding: '4px 8px',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#d1d5db'}`,
            borderRadius: '4px',
            backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
            color: theme === 'dark' ? '#ffffff' : '#374151',
            cursor: currentPage === totalPages ? 'not-allowed' : 'pointer',
            opacity: currentPage === totalPages ? 0.5 : 1,
          }}
        >
          Next
        </button>
      </div>
    </div>
  );
};

// Main DataGrid component
export const DataGrid = <T,>({
  data = [],
  columns,
  loading = false,
  rowHeight = 48,
  headerHeight = 48,
  width,
  height = 400,
  pageSize = 25,
  pagination = false,
  virtualScrolling = true,
  sortable = true,
  filterable = false,
  selectable = false,
  multiSelect = true,
  selectedRowKeys = [],
  rowKey = 'id',
  onRowSelect,
  onRowClick,
  onSort,
  onFilter,
  emptyText = 'No data available',
  className = '',
  rowClassName,
  style,
}: DataGridProps<T>) => {
  const theme = useAppSelector(state => state.dashboard.config.theme);
  
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<SortConfig | null>(null);
  const [filters, setFilters] = useState<FilterConfig>({});
  const [internalSelectedKeys, setInternalSelectedKeys] = useState<Set<string | number>>(
    new Set(selectedRowKeys)
  );

  // Update internal selected keys when prop changes
  useEffect(() => {
    setInternalSelectedKeys(new Set(selectedRowKeys));
  }, [selectedRowKeys]);

  // Get row key function
  const getRowKey = useCallback((record: T): string | number => {
    return typeof rowKey === 'function' ? rowKey(record) : record[rowKey as keyof T] as string | number;
  }, [rowKey]);

  // Filter data
  const filteredData = useMemo(() => {
    let filtered = [...data];
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value === '' || value == null) return;
      
      const column = columns.find(col => col.key === key);
      if (!column) return;
      
      filtered = filtered.filter(record => {
        const recordValue = record[key as keyof T];
        
        if (column.filter?.type === 'number') {
          return Number(recordValue) === Number(value);
        }
        
        return String(recordValue || '').toLowerCase().includes(String(value).toLowerCase());
      });
    });
    
    return filtered;
  }, [data, filters, columns]);

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortConfig) return filteredData;
    
    const column = columns.find(col => col.key === sortConfig.key);
    if (!column) return filteredData;
    
    const sorted = [...filteredData].sort((a, b) => {
      if (column.sorter) {
        return column.sorter(a, b);
      }
      
      const aVal = a[sortConfig.key as keyof T];
      const bVal = b[sortConfig.key as keyof T];
      
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      
      if (aVal < bVal) return -1;
      if (aVal > bVal) return 1;
      return 0;
    });
    
    return sortConfig.direction === 'desc' ? sorted.reverse() : sorted;
  }, [filteredData, sortConfig, columns]);

  // Paginate data
  const paginatedData = useMemo(() => {
    if (!pagination) return sortedData;
    
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return sortedData.slice(startIndex, endIndex);
  }, [sortedData, pagination, currentPage, pageSize]);

  // Get display data
  const displayData = pagination ? paginatedData : sortedData;

  // Event handlers
  const handleSort = useCallback((key: string) => {
    const newSortConfig: SortConfig = {
      key,
      direction: sortConfig?.key === key && sortConfig.direction === 'asc' ? 'desc' : 'asc',
    };
    setSortConfig(newSortConfig);
    onSort?.(newSortConfig);
  }, [sortConfig, onSort]);

  const handleFilter = useCallback((key: string, value: any) => {
    const newFilters = { ...filters, [key]: value };
    if (value === '' || value == null) {
      delete newFilters[key];
    }
    setFilters(newFilters);
    setCurrentPage(1);
    onFilter?.(newFilters);
  }, [filters, onFilter]);

  const handleRowSelect = useCallback((key: string | number, selected: boolean) => {
    const newSelectedKeys = new Set(internalSelectedKeys);
    
    if (selected) {
      if (multiSelect) {
        newSelectedKeys.add(key);
      } else {
        newSelectedKeys.clear();
        newSelectedKeys.add(key);
      }
    } else {
      newSelectedKeys.delete(key);
    }
    
    setInternalSelectedKeys(newSelectedKeys);
    
    const selectedRows = displayData.filter(record => 
      newSelectedKeys.has(getRowKey(record))
    );
    
    onRowSelect?.(selectedRows, Array.from(newSelectedKeys));
  }, [internalSelectedKeys, multiSelect, displayData, getRowKey, onRowSelect]);

  const handleSelectAll = useCallback((selected: boolean) => {
    const newSelectedKeys = new Set<string | number>();
    
    if (selected) {
      displayData.forEach(record => {
        newSelectedKeys.add(getRowKey(record));
      });
    }
    
    setInternalSelectedKeys(newSelectedKeys);
    
    const selectedRows = selected ? displayData : [];
    onRowSelect?.(selectedRows, Array.from(newSelectedKeys));
  }, [displayData, getRowKey, onRowSelect]);

  // Loading state
  if (loading) {
    return (
      <div
        style={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          borderRadius: '8px',
          color: theme === 'dark' ? '#ffffff' : '#374151',
          ...style,
        }}
        className={className}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '18px', marginBottom: '8px' }}>⏳</div>
          <div>Loading data...</div>
        </div>
      </div>
    );
  }

  // Empty state
  if (displayData.length === 0) {
    return (
      <div
        style={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
          border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
          borderRadius: '8px',
          color: theme === 'dark' ? '#9ca3af' : '#6b7280',
          ...style,
        }}
        className={className}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>📭</div>
          <div>{emptyText}</div>
        </div>
      </div>
    );
  }

  const GridComponent = virtualScrolling ? List : 'div';

  return (
    <div
      style={{
        width,
        height,
        border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
        borderRadius: '8px',
        backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        ...style,
      }}
      className={className}
    >
      {/* Header */}
      <DataGridHeader
        columns={columns}
        sortConfig={sortConfig}
        onSort={handleSort}
        onFilter={handleFilter}
        filters={filters}
        theme={theme}
        selectable={selectable}
        selectedRowKeys={internalSelectedKeys}
        totalRows={displayData.length}
        onSelectAll={handleSelectAll}
      />

      {/* Data rows */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {virtualScrolling ? (
          <AutoSizer>
            {({ width: containerWidth, height: containerHeight }) => (
              <List
                width={containerWidth}
                height={containerHeight}
                itemCount={displayData.length}
                itemSize={rowHeight}
                itemData={{
                  items: displayData,
                  columns,
                  selectedRowKeys: internalSelectedKeys,
                  onRowSelect: handleRowSelect,
                  onRowClick,
                  rowKey: getRowKey,
                  theme,
                  selectable,
                  rowClassName,
                } as any}
              >
                {VirtualRow as any}
              </List>
            )}
          </AutoSizer>
        ) : (
          <div style={{ overflow: 'auto', height: '100%' }}>
            {displayData.map((record, index) => (
              <VirtualRow
                key={getRowKey(record)}
                index={index}
                style={{ height: rowHeight }}
                data={{
                  items: displayData,
                  columns,
                  selectedRowKeys: internalSelectedKeys,
                  onRowSelect: handleRowSelect,
                  onRowClick,
                  rowKey: getRowKey,
                  theme,
                  selectable,
                  rowClassName,
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {pagination && (
        <DataGridPagination
          currentPage={currentPage}
          pageSize={pageSize}
          total={sortedData.length}
          onPageChange={setCurrentPage}
          onPageSizeChange={(size) => {
            setCurrentPage(1);
          }}
          theme={theme}
        />
      )}
    </div>
  );
};

export default DataGrid;