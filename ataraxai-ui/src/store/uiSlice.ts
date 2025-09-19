import { StateCreator } from 'zustand';
import { AppView, AppStatus, RagSource, ModelInfo } from './types';

export interface UISlice {
  currentView: AppView;
  appStatus: AppStatus;
  sidebarCollapsed: boolean;
  ragSources: RagSource[];
  models: ModelInfo[];
  searchQuery: string;
  
  // Actions
  setCurrentView: (view: AppView) => void;
  setAppStatus: (status: AppStatus) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  addRagSource: (path: string) => void;
  removeRagSource: (id: string) => void;
  setSearchQuery: (query: string) => void;
  addModel: (name: string, size: string) => void;
  updateModelProgress: (id: string, progress: number) => void;
  markModelDownloaded: (id: string) => void;
}

export const createUISlice: StateCreator<UISlice> = (set, get) => ({
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

  setCurrentView: (view: AppView) => {
    set({ currentView: view });
  },

  setAppStatus: (status: AppStatus) => {
    set({ appStatus: status });
  },

  toggleSidebar: () => {
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
  },

  setSidebarCollapsed: (collapsed: boolean) => {
    set({ sidebarCollapsed: collapsed });
  },

  addRagSource: (path: string) => {
    const newSource: RagSource = {
      id: Date.now().toString(),
      path,
      type: 'directory',
      addedAt: new Date(),
    };
    
    set((state) => ({
      ragSources: [...state.ragSources, newSource],
    }));
  },

  removeRagSource: (id: string) => {
    set((state) => ({
      ragSources: state.ragSources.filter((source) => source.id !== id),
    }));
  },

  setSearchQuery: (query: string) => {
    set({ searchQuery: query });
  },

  addModel: (name: string, size: string) => {
    const newModel: ModelInfo = {
      id: Date.now().toString(),
      name,
      size,
      isDownloaded: false,
      isDownloading: true,
      downloadProgress: 0,
    };
    
    set((state) => ({
      models: [...state.models, newModel],
    }));
  },

  updateModelProgress: (id: string, progress: number) => {
    set((state) => ({
      models: state.models.map((model) =>
        model.id === id
          ? { ...model, downloadProgress: progress }
          : model
      ),
    }));
  },

  markModelDownloaded: (id: string) => {
    set((state) => ({
      models: state.models.map((model) =>
        model.id === id
          ? { ...model, isDownloaded: true, isDownloading: false, downloadProgress: 100 }
          : model
      ),
    }));
  },
});