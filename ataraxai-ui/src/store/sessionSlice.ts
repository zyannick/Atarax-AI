import { StateCreator } from 'zustand';
import { ChatSession } from './types';

export interface SessionSlice {
  sessions: ChatSession[];
  selectedSessionId: string | null;
  
  // Actions
  addSession: (projectId: string, title?: string) => void;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;
  selectSession: (id: string) => void;
  getSessionsByProject: (projectId: string) => ChatSession[];
}

export const createSessionSlice: StateCreator<SessionSlice> = (set, get) => ({
  sessions: [
    {
      id: '1',
      projectId: '1',
      title: 'New Chat',
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ],
  selectedSessionId: '1',

  addSession: (projectId: string, title = 'New Chat') => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      projectId,
      title,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    
    set((state) => ({
      sessions: [...state.sessions, newSession],
      selectedSessionId: newSession.id,
    }));
  },

  renameSession: (id: string, title: string) => {
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id
          ? { ...session, title, updatedAt: new Date() }
          : session
      ),
    }));
  },

  deleteSession: (id: string) => {
    set((state) => {
      const filteredSessions = state.sessions.filter((session) => session.id !== id);
      const newSelectedId = state.selectedSessionId === id 
        ? (filteredSessions[0]?.id || null)
        : state.selectedSessionId;
      
      return {
        sessions: filteredSessions,
        selectedSessionId: newSelectedId,
      };
    });
  },

  selectSession: (id: string) => {
    set({ selectedSessionId: id });
  },

  getSessionsByProject: (projectId: string) => {
    return get().sessions.filter((session) => session.projectId === projectId);
  },
});