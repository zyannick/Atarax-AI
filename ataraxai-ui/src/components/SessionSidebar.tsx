import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { BrainCircuit, Folder, MessageSquare, Plus, Send, X, Settings, Sun, Moon } from 'lucide-react';

const SessionSidebar = () => {
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

    const handleCreateSession = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newSessionTitle.trim() || !selectedProjectId || isCreating) return;

        setIsCreating(true);
        await createSession(selectedProjectId, newSessionTitle);
        setNewSessionTitle('');
        setIsCreating(false);
    };

    if (!selectedProjectId) {
        return (
            <div className="w-72 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700/50 flex flex-col items-center justify-center text-center p-4">
                <Folder size={48} className="text-gray-300 dark:text-gray-600 mb-4" />
                <h3 className="font-semibold text-lg">No Project Selected</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Select a project to view its sessions.</p>
            </div>
        );
    }

    return (
        <div className="w-72 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700/50 flex flex-col p-3">
            <h2 className="text-lg font-semibold mb-4 px-2">Sessions</h2>
            <div className="flex-1 space-y-1 overflow-y-auto pr-1">
                {sessions.map((session) => (
                    <button
                        key={session.session_id}
                        onClick={() => selectSession(session.session_id)}
                        className={`w-full text-left p-2.5 rounded-lg transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                            selectedSessionId === session.session_id
                                ? 'bg-gray-200 dark:bg-gray-700 font-semibold'
                                : 'hover:bg-gray-100 dark:hover:bg-gray-700/50'
                        }`}
                    >
                        <span className="truncate">{session.title}</span>
                    </button>
                ))}
                {isLoading && sessions.length === 0 && <p className="text-sm text-center text-gray-400 py-4">Loading sessions...</p>}
            </div>
            <form onSubmit={handleCreateSession} className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700/50">
                <div className="relative">
                    <input
                        type="text"
                        value={newSessionTitle}
                        onChange={(e) => setNewSessionTitle(e.target.value)}
                        placeholder="New session title..."
                        disabled={isCreating}
                        className="w-full pl-3 pr-10 py-2 rounded-lg border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition disabled:opacity-50"
                    />
                    <button 
                        type="submit"
                        disabled={!newSessionTitle.trim() || isCreating}
                        className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 text-gray-400 hover:text-blue-500 disabled:text-gray-300 dark:disabled:text-gray-500 rounded-md transition"
                    >
                        <Plus size={20} />
                    </button>
                </div>
            </form>
        </div>
    );
};


export default SessionSidebar;