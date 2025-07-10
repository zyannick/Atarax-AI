import React, { useEffect } from 'react';
import { ProjectSidebar } from './components/ProjectSidebar';
import { SessionSidebar } from './components/SessionSidebar';
import { ChatWindow } from './components/ChatWindow';
import { useAppStore } from './store';

export const WorkspaceView = () => {
  const { error } = useAppStore();

  return (
    <div className="flex flex-1 h-full">
      <ProjectSidebar />
      <SessionSidebar />
      <ChatWindow />
      
      {/* Global Error Toast (optional) */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
          <p className="text-sm">{error}</p>
        </div>
      )}
    </div>
  );
};