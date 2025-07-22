/**
 * Real-time Status Hook
 * Provides system status and notifications
 */

import { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../stores';

export interface SystemStatus {
  overall: 'healthy' | 'warning' | 'error' | 'unknown';
  cpu: number;
  memory: number;
  latency: number;
  connections: number;
  performance?: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
}

export interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const useRealTimeStatus = () => {
  const connectionStatus = useSelector((state: RootState) => state.realtime.connectionStatus);
  const lastUpdate = useSelector((state: RootState) => state.realtime.lastUpdate);
  
  const [systemStatus, setSystemStatus] = useState<{ overall: SystemStatus['overall']; performance?: SystemStatus['performance'] }>({
    overall: 'unknown',
    performance: {
      cpu: 0,
      memory: 0,
      disk: 0,
      network: 0,
    },
  });
  
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: '1',
      type: 'info',
      title: 'Welcome to LLMKG Dashboard',
      message: 'Your brain-inspired knowledge graph system is ready.',
      timestamp: new Date(),
      read: false,
    },
  ]);

  // Simulate real-time status updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate CPU/Memory fluctuations
      const cpu = Math.floor(Math.random() * 30 + 20); // 20-50%
      const memory = Math.floor(Math.random() * 20 + 40); // 40-60%
      const disk = Math.floor(Math.random() * 10 + 30); // 30-40%
      const network = Math.floor(Math.random() * 15 + 10); // 10-25%
      
      setSystemStatus({
        overall: connectionStatus === 'connected' ? 'healthy' : 'warning',
        performance: { cpu, memory, disk, network },
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [connectionStatus]);

  // Add notification when connection status changes
  useEffect(() => {
    if (connectionStatus === 'connected') {
      const notification: Notification = {
        id: Date.now().toString(),
        type: 'success',
        title: 'Connected',
        message: 'Real-time data connection established.',
        timestamp: new Date(),
        read: false,
      };
      setNotifications(prev => [notification, ...prev]);
    } else if (connectionStatus === 'disconnected') {
      const notification: Notification = {
        id: Date.now().toString(),
        type: 'warning',
        title: 'Disconnected',
        message: 'Real-time data connection lost. Attempting to reconnect...',
        timestamp: new Date(),
        read: false,
      };
      setNotifications(prev => [notification, ...prev]);
    }
  }, [connectionStatus]);

  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(notif =>
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  };

  const clearNotification = (id: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  const clearAllNotifications = () => {
    setNotifications([]);
  };

  return {
    systemStatus,
    notifications,
    markAsRead,
    clearNotification,
    clearAllNotifications,
    connectionStatus,
    lastUpdate,
  };
};