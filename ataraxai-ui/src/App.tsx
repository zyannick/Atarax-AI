import React from 'react';
import { AppProvider, useAppStore } from './store/AppContext';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { ChatView } from './components/ChatView';
import { RAGSettingsView } from './components/RAGSettingsView';
import { ModelManagerView } from './components/ModelManagerView';
import { BenchmarkView } from './components/BenchmarkView';

function AppContent() {
  const { currentView } = useAppStore();

  // Remove dark theme application since we're using light theme
  // useEffect(() => {
  //   document.documentElement.classList.add('dark');
  // }, []);

  const renderMainContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatView />;
      case 'rag-settings':
        return <RAGSettingsView />;
      case 'model-manager':
        return <ModelManagerView />;
      case 'benchmark':
        return <BenchmarkView />;
      default:
        return <ChatView />;
    }
  };

  return (
    <div className="h-screen w-screen bg-background text-foreground overflow-hidden">
      <div className="flex h-full">
        <LeftSidebar />
        {renderMainContent()}
        <RightSidebar />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}