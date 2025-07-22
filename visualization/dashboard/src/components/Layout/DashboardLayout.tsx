import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from '../Navigation/Sidebar';
import { Header } from '../Navigation/Header';
import { Breadcrumb } from '../Navigation/Breadcrumb';
import { useBreakpoint } from '../../hooks/useBreakpoint';

export const DashboardLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { isMobile } = useBreakpoint();

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="h-screen flex overflow-hidden bg-gray-100">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen || !isMobile} 
        onToggle={toggleSidebar}
      />

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Header */}
        <Header onToggleSidebar={toggleSidebar} />

        {/* Breadcrumb */}
        <div className="bg-white border-b border-gray-200 px-4 sm:px-6 lg:px-8 py-3">
          <Breadcrumb />
        </div>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto focus:outline-none">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;