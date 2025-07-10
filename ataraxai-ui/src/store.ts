import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/tauri';

export interface Project {
  project_id: string;
  name: string;
  description?: string;
}

export interface Session {
  session_id: string;
  title: string;
  project_id: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
}

interface AppState {
  projects: Project[];
  selectedProjectId: string | null;
  sessions: Session[];
  selectedSessionId: string | null;
  messages: Message[];
  isLoading: boolean;
  error: string | null;

  fetchProjects: () => Promise<void>;
  createProject: (name: string) => Promise<void>;
  selectProject: (projectId: string | null) => Promise<void>;
  
  fetchSessions: (projectId: string) => Promise<void>;
  createSession: (projectId: string, title: string) => Promise<void>;
  selectSession: (sessionId: string | null) => Promise<void>;

  fetchMessages: (sessionId: string) => Promise<void>;
  sendMessage: (sessionId: string, query: string) => Promise<void>;
}

export const useAppStore = create<AppState>((set, get) => ({
  projects: [],
  selectedProjectId: null,
  sessions: [],
  selectedSessionId: null,
  messages: [],
  isLoading: false,
  error: null,

  fetchProjects: async () => {
    set({ isLoading: true, error: null });
    try {
      const projects = await invoke<Project[]>('list_projects');
      set({ projects });
      if (projects.length > 0 && !get().selectedProjectId) {
        await get().selectProject(projects[0].project_id);
      }
    } catch (e) {
      set({ error: `Failed to fetch projects: ${e}` });
    } finally {
      set({ isLoading: false });
    }
  },

  createProject: async (name: string) => {
    set({ isLoading: true, error: null });
    try {
      await invoke('create_project', { name, description: "New project" });
      await get().fetchProjects();
    } catch (e) {
      set({ error: `Failed to create project: ${e}` });
    } finally {
      set({ isLoading: false });
    }
  },

  selectProject: async (projectId: string | null) => {
    set({ selectedProjectId: projectId, sessions: [], selectedSessionId: null, messages: [] });
    if (projectId) {
      await get().fetchSessions(projectId);
    }
  },

  fetchSessions: async (projectId: string) => {
    set({ isLoading: true, error: null });
    try {
      const sessions = await invoke<Session[]>('list_sessions', { projectId });
      set({ sessions });
    } catch (e) {
      set({ error: `Failed to fetch sessions: ${e}` });
    } finally {
      set({ isLoading: false });
    }
  },

  createSession: async (projectId: string, title: string) => {
     set({ isLoading: true, error: null });
    try {
      await invoke('create_session', { projectId, title });
      await get().fetchSessions(projectId);
    } catch (e) {
       set({ error: `Failed to create session: ${e}` });
    } finally {
       set({ isLoading: false });
    }
  },

  selectSession: async (sessionId: string | null) => {
    set({ selectedSessionId: sessionId, messages: [] });
    if (sessionId) {
      await get().fetchMessages(sessionId);
    }
  },

  fetchMessages: async (sessionId: string) => {
    set({ isLoading: true, error: null });
    try {
        const messages = await invoke<Message[]>('list_messages', { sessionId });
        set({ messages });
    } catch (e) {
        set({ error: `Failed to fetch messages: ${e}` });
    } finally {
        set({ isLoading: false });
    }
  },

  sendMessage: async (sessionId: string, query: string) => {
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content: query };
    set(state => ({ messages: [...state.messages, userMessage] }));
    
    try {
        const assistantResponse = await invoke<Message>('send_message', { sessionId, userQuery: query });
        set(state => ({ messages: [...state.messages, assistantResponse] }));
    } catch (e) {
        const errorMsg: Message = { id: crypto.randomUUID(), role: 'error', content: `Failed to get response: ${e}`};
        set(state => ({ messages: [...state.messages, errorMsg] }));
    }
  }
}));