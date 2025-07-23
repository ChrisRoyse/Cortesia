import { useState, useEffect } from 'react';
import { getAppConfig, AppInitConfig } from '../config/initialization';

export function useAppConfig(): AppInitConfig | null {
  const [config, setConfig] = useState<AppInitConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate async config loading
    const loadConfig = async () => {
      try {
        // In a real app, this might fetch config from an API
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const appConfig = getAppConfig();
        setConfig(appConfig);
      } catch (error) {
        console.error('Failed to load app config:', error);
        // Use default config as fallback
        setConfig(getAppConfig());
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  return loading ? null : config;
}