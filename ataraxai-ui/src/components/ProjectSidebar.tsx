import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store';

export const ProjectSidebar = () => {
  const { 
    projects, 
    selectedProjectId, 
    selectProject, 
    createProject, 
    fetchProjects,
    isLoading,
    error 
  } = useAppStore();
  const [newProjectName, setNewProjectName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreateProject = async () => {
    if (!newProjectName.trim() || isCreating) return;
    
    setIsCreating(true);
    try {
      await createProject(newProjectName);
      setNewProjectName('');
    } catch (error) {
      // Error is handled in the store
    } finally {
      setIsCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCreateProject();
    }
  };

  return (
    <div className="w-64 bg-gray-50 dark:bg-gray-800/50 border-r border-gray-200 dark:border-gray-700 p-4 flex flex-col">
      <h2 className="text-lg font-semibold mb-4">Projects</h2>
      
      {/* Loading State */}
      {isLoading && projects.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
        </div>
      )}
      
      {/* Projects List */}
      <div className="flex-1 space-y-2 overflow-y-auto">
        {projects.length === 0 && !isLoading ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-2xl mb-2">📁</div>
            <p className="text-sm">No projects yet</p>
            <p className="text-xs mt-1">Create your first project below</p>
          </div>
        ) : (
          projects.map((project) => (
            <button
              key={project.project_id}
              onClick={() => selectProject(project.project_id)}
              className={`w-full text-left p-2 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                selectedProjectId === project.project_id 
                  ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300' 
                  : 'hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
              aria-label={`Select project ${project.name}`}
            >
              <div className="truncate">{project.name}</div>
              {project.description && (
                <div className="text-xs text-gray-500 dark:text-gray-400 truncate mt-1">
                  {project.description}
                </div>
              )}
            </button>
          ))
        )}
      </div>
      
      {/* Create Project Form */}
      <div className="mt-4">
        <input
          type="text"
          value={newProjectName}
          onChange={(e) => setNewProjectName(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="New project name..."
          disabled={isCreating}
          className="w-full p-2 rounded-md border-2 border-gray-300 dark:border-gray-600 dark:bg-gray-800 mb-2 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          aria-label="New project name"
        />
        <button 
          onClick={handleCreateProject}
          disabled={!newProjectName.trim() || isCreating}
          className="w-full p-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label="Create new project"
        >
          {isCreating ? 'Creating...' : 'Create Project'}
        </button>
      </div>
    </div>
  );
};