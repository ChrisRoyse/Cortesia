import React from 'react';
import { MemoryDashboard } from '../src';

function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <MemoryDashboard 
        wsUrl="ws://localhost:8080" 
        className="w-full"
      />
    </div>
  );
}

export default App;