import React from 'react';
import {
  ClockIcon,
  UserCircleIcon,
  CpuChipIcon,
  DatabaseIcon,
  WrenchScrewdriverIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

export interface Activity {
  id: string;
  title: string;
  description: string;
  type: 'system' | 'user' | 'error' | 'success' | 'warning' | 'info';
  timestamp: Date;
  user?: string;
  component?: string;
  metadata?: Record<string, any>;
}

interface ActivityFeedProps {
  activities: Activity[];
  className?: string;
  showTimestamps?: boolean;
  maxItems?: number;
}

export const ActivityFeed: React.FC<ActivityFeedProps> = ({
  activities,
  className = "",
  showTimestamps = true,
  maxItems,
}) => {
  const displayActivities = maxItems ? activities.slice(0, maxItems) : activities;

  const getActivityIcon = (activity: Activity) => {
    switch (activity.type) {
      case 'user':
        return <UserCircleIcon className="w-5 h-5 text-blue-500" />;
      case 'system':
        return <CpuChipIcon className="w-5 h-5 text-purple-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      case 'success':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return <InformationCircleIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getActivityBgColor = (type: Activity['type']) => {
    switch (type) {
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'user':
        return 'bg-blue-50 border-blue-200';
      case 'system':
        return 'bg-purple-50 border-purple-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const formatRelativeTime = (timestamp: Date) => {
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - timestamp.getTime()) / 1000);

    if (diffInSeconds < 60) {
      return 'just now';
    }
    
    const diffInMinutes = Math.floor(diffInSeconds / 60);
    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    }
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    }
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  if (displayActivities.length === 0) {
    return (
      <div className={`text-center py-8 ${className}`}>
        <ClockIcon className="w-8 h-8 text-gray-300 mx-auto mb-4" />
        <p className="text-gray-500">No recent activity</p>
      </div>
    );
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {displayActivities.map((activity, index) => (
        <div
          key={activity.id}
          className="relative flex items-start space-x-3 p-3 bg-white rounded-lg border border-gray-200 hover:shadow-sm transition-shadow"
        >
          {/* Timeline connector for non-first items */}
          {index > 0 && (
            <div className="absolute left-6 -top-3 w-px h-3 bg-gray-200" />
          )}
          
          {/* Activity icon */}
          <div className="flex-shrink-0">
            {getActivityIcon(activity)}
          </div>
          
          {/* Activity content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-gray-900 truncate">
                {activity.title}
              </h4>
              {showTimestamps && (
                <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
                  {formatRelativeTime(activity.timestamp)}
                </span>
              )}
            </div>
            
            <p className="text-sm text-gray-600 mt-1">
              {activity.description}
            </p>
            
            {/* Additional metadata */}
            <div className="flex items-center mt-2 space-x-4">
              {activity.user && (
                <span className="text-xs text-gray-500">
                  by {activity.user}
                </span>
              )}
              {activity.component && (
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  {activity.component}
                </span>
              )}
            </div>
            
            {/* Metadata details (if any) */}
            {activity.metadata && Object.keys(activity.metadata).length > 0 && (
              <div className="mt-2 p-2 bg-gray-50 rounded text-xs">
                {Object.entries(activity.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-500">{key}:</span>
                    <span className="text-gray-700">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      
      {maxItems && activities.length > maxItems && (
        <div className="text-center py-2">
          <span className="text-sm text-gray-500">
            Showing {maxItems} of {activities.length} activities
          </span>
        </div>
      )}
    </div>
  );
};

export default ActivityFeed;