import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Bars3Icon,
  BellIcon,
  UserCircleIcon,
  MagnifyingGlassIcon,
  Cog6ToothIcon,
  QuestionMarkCircleIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline';
import { useRealTimeStatus } from '../../hooks/useRealTimeStatus';
import { StatusIndicator } from '../common/StatusIndicator';
import { DropdownMenu } from '../common/DropdownMenu';
import { NotificationCenter } from '../common/NotificationCenter';
import { SearchBox } from '../common/SearchBox';
import { useOnClickOutside } from '../../hooks/useOnClickOutside';

interface HeaderProps {
  onToggleSidebar: () => void;
  title?: string;
}

export const Header: React.FC<HeaderProps> = ({ onToggleSidebar, title }) => {
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const searchRef = useRef<HTMLDivElement>(null);
  const notificationsRef = useRef<HTMLDivElement>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  const { systemStatus, notifications } = useRealTimeStatus();

  useOnClickOutside(searchRef, () => setIsSearchExpanded(false));
  useOnClickOutside(notificationsRef, () => setIsNotificationsOpen(false));
  useOnClickOutside(profileMenuRef, () => setIsProfileMenuOpen(false));

  const unreadNotifications = notifications.filter(n => !n.read).length;

  const profileMenuItems = [
    {
      id: 'profile',
      label: 'Profile',
      icon: UserCircleIcon,
      onClick: () => console.log('Profile clicked'),
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Cog6ToothIcon,
      onClick: () => console.log('Settings clicked'),
    },
    {
      id: 'help',
      label: 'Help & Support',
      icon: QuestionMarkCircleIcon,
      onClick: () => console.log('Help clicked'),
    },
    {
      type: 'divider' as const,
    },
    {
      id: 'logout',
      label: 'Sign out',
      icon: ArrowRightOnRectangleIcon,
      onClick: () => console.log('Logout clicked'),
      className: 'text-red-600 hover:text-red-700',
    },
  ];

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left section */}
          <div className="flex items-center space-x-4">
            {/* Sidebar toggle */}
            <button
              onClick={onToggleSidebar}
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 lg:hidden"
              aria-label="Toggle sidebar"
            >
              <Bars3Icon className="w-6 h-6" />
            </button>

            {/* Logo and title */}
            <div className="flex items-center space-x-3">
              <Link to="/" className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">LK</span>
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-xl font-bold text-gray-900">
                    {title || 'LLMKG Dashboard'}
                  </h1>
                  <p className="text-xs text-gray-500">Brain-inspired Knowledge Graph</p>
                </div>
              </Link>
            </div>

            {/* System status */}
            <div className="hidden md:flex items-center space-x-2">
              <StatusIndicator 
                status={systemStatus.overall} 
                showLabel 
                className="text-sm"
              />
            </div>
          </div>

          {/* Right section */}
          <div className="flex items-center space-x-2">
            {/* Search */}
            <div ref={searchRef} className="relative">
              {isSearchExpanded ? (
                <div className="absolute right-0 top-0 w-80 z-50">
                  <SearchBox
                    value={searchQuery}
                    onChange={setSearchQuery}
                    placeholder="Search features, tools, or content..."
                    autoFocus
                    className="w-full"
                    onEscape={() => setIsSearchExpanded(false)}
                  />
                </div>
              ) : (
                <button
                  onClick={() => setIsSearchExpanded(true)}
                  className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  aria-label="Search"
                >
                  <MagnifyingGlassIcon className="w-5 h-5" />
                </button>
              )}
            </div>

            {/* Notifications */}
            <div ref={notificationsRef} className="relative">
              <button
                onClick={() => setIsNotificationsOpen(!isNotificationsOpen)}
                className="relative p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label={`Notifications ${unreadNotifications > 0 ? `(${unreadNotifications} unread)` : ''}`}
              >
                <BellIcon className="w-5 h-5" />
                {unreadNotifications > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {unreadNotifications > 9 ? '9+' : unreadNotifications}
                  </span>
                )}
              </button>

              {isNotificationsOpen && (
                <div className="absolute right-0 mt-2 w-96 z-50">
                  <NotificationCenter
                    notifications={notifications}
                    onClose={() => setIsNotificationsOpen(false)}
                  />
                </div>
              )}
            </div>

            {/* Performance indicator */}
            <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 bg-gray-50 rounded-lg">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-xs text-gray-600">
                {systemStatus.performance?.cpu || 0}% CPU
              </span>
            </div>

            {/* Profile menu */}
            <div ref={profileMenuRef} className="relative">
              <button
                onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                className="flex items-center space-x-2 p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="User menu"
              >
                <UserCircleIcon className="w-6 h-6" />
                <span className="hidden md:block text-sm font-medium text-gray-700">
                  Admin User
                </span>
              </button>

              {isProfileMenuOpen && (
                <div className="absolute right-0 mt-2 w-56 z-50">
                  <DropdownMenu
                    items={profileMenuItems}
                    onClose={() => setIsProfileMenuOpen(false)}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile search overlay */}
      {isSearchExpanded && (
        <div className="lg:hidden bg-white border-t border-gray-200 p-4">
          <SearchBox
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search features, tools, or content..."
            autoFocus
            className="w-full"
            onEscape={() => setIsSearchExpanded(false)}
          />
        </div>
      )}

      {/* System status bar for mobile */}
      <div className="md:hidden bg-gray-50 border-t border-gray-200 px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <StatusIndicator status={systemStatus.overall} />
            <span className="text-sm text-gray-600">System Status</span>
          </div>
          <div className="text-xs text-gray-500">
            CPU: {systemStatus.performance?.cpu || 0}% | 
            Memory: {systemStatus.performance?.memory || 0}%
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;