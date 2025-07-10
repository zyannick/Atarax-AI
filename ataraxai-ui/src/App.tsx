import React, { useState, useEffect } from 'react';
import { LayoutDashboard, BrainCircuit, Settings as SettingsIcon } from 'lucide-react';
import { WorkspaceView } from './WorkspaceView';

function App()
{
  const [activeView, setActiveView] = useState('workspace'); // 'workspace', 'knowledge', or 'settings'

  const renderActiveView = () => {
    switch (activeView) {
      case 'knowledge':
        return <div className="flex items-center justify-center h-full">
          <BrainCircuit className="w-16 h-16 text-gray-500" />
          <p className="text-gray-500 ml-4">Knowledge Base is under construction...</p>
        </div>;
      case 'settings':
        return <div className="flex items-center justify-center h-full">
          <BrainCircuit className="w-16 h-16 text-gray-500" />
          <p className="text-gray-500 ml-4">Settings Base is under construction...</p>
        </div>;
      case 'workspace':
      default:
        return <WorkspaceView />;
    }
  };

  return (
    <> {renderActiveView()} </>
  );
}

export default App;