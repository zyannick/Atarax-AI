import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { Project, ChatSession, Message, AppView, AppStatus, RagSource, ModelInfo, BenchmarkTest, BenchmarkResult, BenchmarkSession } from './types';

interface AppState {
  // Project state
  projects: Project[];
  selectedProjectId: string | null;
  
  // Session state
  sessions: ChatSession[];
  selectedSessionId: string | null;
  
  // Message state
  messages: Message[];
  isTyping: boolean;
  
  // UI state
  currentView: AppView;
  appStatus: AppStatus;
  sidebarCollapsed: boolean;
  ragSources: RagSource[];
  models: ModelInfo[];
  searchQuery: string;
  
  // Benchmark state
  benchmarkTests: BenchmarkTest[];
  benchmarkSessions: BenchmarkSession[];
  activeBenchmarkId: string | null;
}

type AppAction = 
  | { type: 'SET_CURRENT_VIEW'; payload: AppView }
  | { type: 'SET_APP_STATUS'; payload: AppStatus }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_SIDEBAR_COLLAPSED'; payload: boolean }
  | { type: 'SELECT_PROJECT'; payload: string }
  | { type: 'SELECT_SESSION'; payload: string }
  | { type: 'ADD_PROJECT'; payload: { name: string } }
  | { type: 'RENAME_PROJECT'; payload: { id: string; name: string } }
  | { type: 'DELETE_PROJECT'; payload: { id: string } }
  | { type: 'ADD_SESSION'; payload: { projectId: string; title?: string } }
  | { type: 'ADD_MESSAGE'; payload: { sessionId: string; content: string; role: 'user' | 'assistant'; type?: 'text' | 'image' | 'voice' | 'video'; metadata?: any } }
  | { type: 'SET_TYPING'; payload: boolean }
  | { type: 'ADD_RAG_SOURCE'; payload: { path: string } }
  | { type: 'REMOVE_RAG_SOURCE'; payload: { id: string } }
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'ADD_MODEL'; payload: { name: string; size: string } }
  | { type: 'UPDATE_MODEL_PROGRESS'; payload: { id: string; progress: number } }
  | { type: 'MARK_MODEL_DOWNLOADED'; payload: { id: string } }
  | { type: 'ADD_BENCHMARK_TEST'; payload: { name: string; description: string; category: string; estimatedDuration: number } }
  | { type: 'ADD_BENCHMARK_SESSION'; payload: { name: string; modelIds: string[]; testIds: string[] } }
  | { type: 'SET_ACTIVE_BENCHMARK'; payload: { id: string } };

const initialState: AppState = {
  projects: [
    {
      id: '1',
      name: 'General',
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ],
  selectedProjectId: '1',
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
  messages: [
    {
      id: '1',
      sessionId: '1',
      content: 'Hello! I\'m Atarax-AI, your local AI assistant. How can I help you today?',
      role: 'assistant',
      timestamp: new Date(),
      type: 'text',
    },
  ],
  isTyping: false,
  currentView: 'chat',
  appStatus: 'unlocked',
  sidebarCollapsed: false,
  ragSources: [
    {
      id: '1',
      path: '/Users/developer/Documents/Projects',
      type: 'directory',
      addedAt: new Date(),
    },
    {
      id: '2',
      path: '/Users/developer/Knowledge Base',
      type: 'directory',
      addedAt: new Date(),
    },
  ],
  models: [
    {
      id: '1',
      name: 'llama2-7b-chat',
      size: '3.8 GB',
      isDownloaded: true,
      isDownloading: false,
    },
    {
      id: '2',
      name: 'mistral-7b-instruct',
      size: '4.1 GB',
      isDownloaded: false,
      isDownloading: true,
      downloadProgress: 68,
    },
  ],
  searchQuery: '',
  benchmarkTests: [
    {
      id: '1',
      name: 'Text Generation Speed',
      description: 'Measures tokens per second for text generation tasks',
      category: 'performance',
      estimatedDuration: 60,
    },
    {
      id: '2',
      name: 'Reasoning Accuracy',
      description: 'Tests logical reasoning capabilities with standardized questions',
      category: 'reasoning',
      estimatedDuration: 180,
    },
    {
      id: '3',
      name: 'Memory Efficiency',
      description: 'Evaluates memory usage and optimization during inference',
      category: 'memory',
      estimatedDuration: 120,
    },
    {
      id: '4',
      name: 'Code Generation',
      description: 'Tests programming task completion and code quality',
      category: 'accuracy',
      estimatedDuration: 240,
    },
    {
      id: '5',
      name: 'Mathematical Problem Solving',
      description: 'Evaluates mathematical reasoning and calculation accuracy',
      category: 'reasoning',
      estimatedDuration: 150,
    },
  ],
  benchmarkSessions: [],
  activeBenchmarkId: null,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_CURRENT_VIEW':
      return { ...state, currentView: action.payload };
      
    case 'SET_APP_STATUS':
      return { ...state, appStatus: action.payload };
      
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
      
    case 'SET_SIDEBAR_COLLAPSED':
      return { ...state, sidebarCollapsed: action.payload };
      
    case 'SELECT_PROJECT':
      return { ...state, selectedProjectId: action.payload };
      
    case 'SELECT_SESSION':
      return { ...state, selectedSessionId: action.payload };
      
    case 'ADD_PROJECT': {
      const newProject: Project = {
        id: Date.now().toString(),
        name: action.payload.name,
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      return {
        ...state,
        projects: [...state.projects, newProject],
      };
    }
    
    case 'RENAME_PROJECT': {
      const updatedProjects = state.projects.map(project =>
        project.id === action.payload.id
          ? { ...project, name: action.payload.name }
          : project
      );
      return {
        ...state,
        projects: updatedProjects,
      };
    }
    
    case 'DELETE_PROJECT': {
      const filteredProjects = state.projects.filter(project => project.id !== action.payload.id);
      // If we're deleting the selected project, select the first remaining project
      const newSelectedProjectId = state.selectedProjectId === action.payload.id 
        ? (filteredProjects.length > 0 ? filteredProjects[0].id : null)
        : state.selectedProjectId;
      
      // Also remove any sessions from the deleted project
      const filteredSessions = state.sessions.filter(session => session.projectId !== action.payload.id);
      const newSelectedSessionId = state.sessions.some(session => session.id === state.selectedSessionId && session.projectId === action.payload.id)
        ? (filteredSessions.length > 0 ? filteredSessions[0].id : null)
        : state.selectedSessionId;
      
      return {
        ...state,
        projects: filteredProjects,
        selectedProjectId: newSelectedProjectId,
        sessions: filteredSessions,
        selectedSessionId: newSelectedSessionId,
      };
    }
    
    case 'ADD_SESSION': {
      const newSession: ChatSession = {
        id: Date.now().toString(),
        projectId: action.payload.projectId,
        title: action.payload.title || 'New Chat',
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      return {
        ...state,
        sessions: [...state.sessions, newSession],
        selectedSessionId: newSession.id,
      };
    }
    
    case 'ADD_MESSAGE': {
      const newMessage: Message = {
        id: Date.now().toString(),
        sessionId: action.payload.sessionId,
        content: action.payload.content,
        role: action.payload.role,
        timestamp: new Date(),
        type: action.payload.type || 'text',
        metadata: action.payload.metadata,
      };
      return {
        ...state,
        messages: [...state.messages, newMessage],
      };
    }
    
    case 'SET_TYPING':
      return { ...state, isTyping: action.payload };
      
    case 'ADD_RAG_SOURCE': {
      const newSource: RagSource = {
        id: Date.now().toString(),
        path: action.payload.path,
        type: 'directory',
        addedAt: new Date(),
      };
      return {
        ...state,
        ragSources: [...state.ragSources, newSource],
      };
    }
    
    case 'REMOVE_RAG_SOURCE':
      return {
        ...state,
        ragSources: state.ragSources.filter(source => source.id !== action.payload.id),
      };
      
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };
      
    case 'ADD_MODEL': {
      const newModel: ModelInfo = {
        id: Date.now().toString(),
        name: action.payload.name,
        size: action.payload.size,
        isDownloaded: false,
        isDownloading: true,
        downloadProgress: 0,
      };
      return {
        ...state,
        models: [...state.models, newModel],
      };
    }
    
    case 'UPDATE_MODEL_PROGRESS':
      return {
        ...state,
        models: state.models.map(model =>
          model.id === action.payload.id
            ? { ...model, downloadProgress: action.payload.progress }
            : model
        ),
      };
      
    case 'MARK_MODEL_DOWNLOADED':
      return {
        ...state,
        models: state.models.map(model =>
          model.id === action.payload.id
            ? { ...model, isDownloaded: true, isDownloading: false, downloadProgress: 100 }
            : model
        ),
      };
      
    case 'ADD_BENCHMARK_TEST': {
      const newTest: BenchmarkTest = {
        id: Date.now().toString(),
        name: action.payload.name,
        description: action.payload.description,
        category: action.payload.category,
        estimatedDuration: action.payload.estimatedDuration,
      };
      return {
        ...state,
        benchmarkTests: [...state.benchmarkTests, newTest],
      };
    }
    
    case 'ADD_BENCHMARK_SESSION': {
      const newSession: BenchmarkSession = {
        id: Date.now().toString(),
        name: action.payload.name,
        modelIds: action.payload.modelIds,
        testIds: action.payload.testIds,
        status: 'pending',
        progress: 0,
        results: [],
      };
      return {
        ...state,
        benchmarkSessions: [...state.benchmarkSessions, newSession],
        activeBenchmarkId: newSession.id,
      };
    }
    
    case 'SET_ACTIVE_BENCHMARK':
      return { ...state, activeBenchmarkId: action.payload.id };
      
    default:
      return state;
  }
}

interface AppContextValue extends AppState {
  // Actions
  setCurrentView: (view: AppView) => void;
  setAppStatus: (status: AppStatus) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  selectProject: (id: string) => void;
  selectSession: (id: string) => void;
  addProject: (name: string) => void;
  renameProject: (id: string, name: string) => void;
  deleteProject: (id: string) => void;
  addSession: (projectId: string, title?: string) => void;
  addMessage: (sessionId: string, content: string, role: 'user' | 'assistant', type?: 'text' | 'image' | 'voice' | 'video', metadata?: any) => void;
  setTyping: (typing: boolean) => void;
  addRagSource: (path: string) => void;
  removeRagSource: (id: string) => void;
  setSearchQuery: (query: string) => void;
  addModel: (name: string, size: string) => void;
  updateModelProgress: (id: string, progress: number) => void;
  markModelDownloaded: (id: string) => void;
  addBenchmarkTest: (name: string, description: string, category: 'performance' | 'accuracy' | 'reasoning' | 'memory', estimatedDuration: number) => void;
  addBenchmarkSession: (name: string, modelIds: string[], testIds: string[]) => void;
  setActiveBenchmark: (id: string) => void;
  getSessionsByProject: (projectId: string) => ChatSession[];
  getMessagesBySession: (sessionId: string) => Message[];
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const contextValue: AppContextValue = {
    ...state,
    
    setCurrentView: (view: AppView) => dispatch({ type: 'SET_CURRENT_VIEW', payload: view }),
    setAppStatus: (status: AppStatus) => dispatch({ type: 'SET_APP_STATUS', payload: status }),
    toggleSidebar: () => dispatch({ type: 'TOGGLE_SIDEBAR' }),
    setSidebarCollapsed: (collapsed: boolean) => dispatch({ type: 'SET_SIDEBAR_COLLAPSED', payload: collapsed }),
    selectProject: (id: string) => dispatch({ type: 'SELECT_PROJECT', payload: id }),
    selectSession: (id: string) => dispatch({ type: 'SELECT_SESSION', payload: id }),
    addProject: (name: string) => dispatch({ type: 'ADD_PROJECT', payload: { name } }),
    renameProject: (id: string, name: string) => dispatch({ type: 'RENAME_PROJECT', payload: { id, name } }),
    deleteProject: (id: string) => dispatch({ type: 'DELETE_PROJECT', payload: { id } }),
    addSession: (projectId: string, title?: string) => dispatch({ type: 'ADD_SESSION', payload: { projectId, title } }),
    addMessage: (sessionId: string, content: string, role: 'user' | 'assistant', type?: 'text' | 'image' | 'voice' | 'video', metadata?: any) => 
      dispatch({ type: 'ADD_MESSAGE', payload: { sessionId, content, role, type, metadata } }),
    setTyping: (typing: boolean) => dispatch({ type: 'SET_TYPING', payload: typing }),
    addRagSource: (path: string) => dispatch({ type: 'ADD_RAG_SOURCE', payload: { path } }),
    removeRagSource: (id: string) => dispatch({ type: 'REMOVE_RAG_SOURCE', payload: { id } }),
    setSearchQuery: (query: string) => dispatch({ type: 'SET_SEARCH_QUERY', payload: query }),
    addModel: (name: string, size: string) => dispatch({ type: 'ADD_MODEL', payload: { name, size } }),
    updateModelProgress: (id: string, progress: number) => dispatch({ type: 'UPDATE_MODEL_PROGRESS', payload: { id, progress } }),
    markModelDownloaded: (id: string) => dispatch({ type: 'MARK_MODEL_DOWNLOADED', payload: { id } }),
    
    addBenchmarkTest: (name: string, description: string, category: 'performance' | 'accuracy' | 'reasoning' | 'memory', estimatedDuration: number) => 
      dispatch({ type: 'ADD_BENCHMARK_TEST', payload: { name, description, category, estimatedDuration } }),
    addBenchmarkSession: (name: string, modelIds: string[], testIds: string[]) => 
      dispatch({ type: 'ADD_BENCHMARK_SESSION', payload: { name, modelIds, testIds } }),
    setActiveBenchmark: (id: string) => dispatch({ type: 'SET_ACTIVE_BENCHMARK', payload: { id } }),
    
    getSessionsByProject: (projectId: string) => state.sessions.filter(session => session.projectId === projectId),
    getMessagesBySession: (sessionId: string) => state.messages.filter(message => message.sessionId === sessionId),
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppStore(): AppContextValue {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppStore must be used within an AppProvider');
  }
  return context;
}