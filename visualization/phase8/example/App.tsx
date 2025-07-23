import React from 'react';
import { CognitivePatternDashboard } from '../src';

function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <CognitivePatternDashboard 
        wsUrl="ws://localhost:8080" 
        className="w-full"
      />
    </div>
  );
}

export default App;