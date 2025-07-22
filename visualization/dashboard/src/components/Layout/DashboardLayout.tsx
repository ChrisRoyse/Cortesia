import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Box } from '@mui/material';
import { Sidebar } from '../Navigation/Sidebar';
import { Header } from '../Navigation/Header';
import { Breadcrumb } from '../Navigation/Breadcrumb';
import { useBreakpoint } from '../../hooks/useBreakpoint';
import { useTheme } from '../../hooks/useTheme';

export const DashboardLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { isMobile } = useBreakpoint();
  const { colors, spacing } = useTheme();

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        overflow: 'hidden',
        backgroundColor: colors.background.primary,
      }}
    >
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen || !isMobile} 
        onToggle={toggleSidebar}
      />

      {/* Main content */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <Header onToggleSidebar={toggleSidebar} />

        {/* Breadcrumb */}
        <Box
          sx={{
            backgroundColor: colors.surface.primary,
            borderBottom: `1px solid ${colors.border.primary}`,
            px: { xs: 2, sm: 3, lg: 4 },
            py: 1.5,
          }}
        >
          <Breadcrumb />
        </Box>

        {/* Page content */}
        <Box
          component="main"
          role="main"
          sx={{
            flex: 1,
            overflowY: 'auto',
            backgroundColor: colors.background.secondary,
            '&:focus': {
              outline: 'none',
            },
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default DashboardLayout;