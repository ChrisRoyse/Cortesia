import React from 'react';
import { DebuggingDashboard } from '../src';

function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <DebuggingDashboard 
        wsUrl="ws://localhost:8080" 
        className="w-full"
      />
    </div>
  );
}

export default App;