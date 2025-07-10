import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store';

export const SessionSidebar = () => {
  const { 
    sessions, 
    selectedSessionId, 
    selectSession, 
    createSession, 
    selectedProjectId,
    isLoading 
  } = useAppStore();
  const [newSessionTitle, setNewSessionTitle] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateSession = async () => {
    if (!newSessionTitle.trim() || !selectedProjectId || isCreating) return;
    
    setIsCreating(true);
    try {
      await createSession(selectedProjectId, newSessionTitle);
      setNewSessionTitle('');
    } catch (error) {
    } finally {
      setIsCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreateSession();
    }
  };

  if (!selectedProjectId) {
    return (
      <div className="w-72 bg-white dark:bg-gray-900 p-4 border-r dark:border-gray-700 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">👈</div>
          <h2 className="text-lg font-semibold mb-2">No Project Selected</h2>
          <p className="text-sm">Select a project to view sessions</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-72 bg-white dark:bg-gray-900 p-4 border-r dark:border-gray-700 flex flex-col">
      <h2 className="text-lg font-semibold mb-4">Chat Sessions</h2>
      
      {/* Loading State */}
      {isLoading && sessions.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-500"></div>
        </div>
      )}
      
      {/* Sessions List */}
      <div className="flex-1 space-y-2 overflow-y-auto">
        {sessions.length === 0 && !isLoading ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-2xl mb-2">💬</div>
            <p className="text-sm">No chat sessions yet</p>
            <p className="text-xs mt-1">Create your first session below</p>
          </div>
        ) : (
          sessions.map((session) => (
            <button
              key={session.session_id}
              onClick={() => selectSession(session.session_id)}
              className={`w-full text-left p-2 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 ${
                selectedSessionId === session.session_id 
                  ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300' 
                  : 'hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
              aria-label={`Select session ${session.title}`}
            >
              <div className="truncate">{session.title}</div>
            </button>
          ))
        )}
      </div>
      
      {/* Create Session Form */}
      <div className="mt-4">
        <input
          type="text"
          value={newSessionTitle}
          onChange={(e) => setNewSessionTitle(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="New session title..."
          disabled={isCreating}
          className="w-full p-2 rounded-md border-2 border-gray-300 dark:border-gray-600 dark:bg-gray-800 mb-2 focus:outline-none focus:border-green-500 disabled:opacity-50"
          aria-label="New session title"
        />
        <button 
          onClick={handleCreateSession} 
          disabled={!newSessionTitle.trim() || isCreating}
          className="w-full p-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label="Create new session"
        >
          {isCreating ? 'Creating...' : 'New Chat'}
        </button>
      </div>
    </div>
  );
};