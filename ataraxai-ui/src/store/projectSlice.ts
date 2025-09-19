import { StateCreator } from 'zustand';
import { Project } from './types';

export interface ProjectSlice {
  projects: Project[];
  selectedProjectId: string | null;
  
  // Actions
  addProject: (name: string) => void;
  renameProject: (id: string, name: string) => void;
  deleteProject: (id: string) => void;
  selectProject: (id: string) => void;
}

export const createProjectSlice: StateCreator<ProjectSlice> = (set, get) => ({
  projects: [
    {
      id: '1',
      name: 'General',
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ],
  selectedProjectId: '1',

  addProject: (name: string) => {
    const newProject: Project = {
      id: Date.now().toString(),
      name,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    
    set((state) => ({
      projects: [...state.projects, newProject],
    }));
  },

  renameProject: (id: string, name: string) => {
    set((state) => ({
      projects: state.projects.map((project) =>
        project.id === id
          ? { ...project, name, updatedAt: new Date() }
          : project
      ),
    }));
  },

  deleteProject: (id: string) => {
    set((state) => {
      const filteredProjects = state.projects.filter((project) => project.id !== id);
      const newSelectedId = state.selectedProjectId === id 
        ? (filteredProjects[0]?.id || null)
        : state.selectedProjectId;
      
      return {
        projects: filteredProjects,
        selectedProjectId: newSelectedId,
      };
    });
  },

  selectProject: (id: string) => {
    set({ selectedProjectId: id });
  },
});