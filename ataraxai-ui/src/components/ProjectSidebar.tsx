import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store'; // Assuming store.ts is in the same directory
import { BrainCircuit, Folder, MessageSquare, Plus, Send, X, Settings, Sun, Moon } from 'lucide-react';



const ProjectSidebar = () => {
  const { projects, selectedProjectId, selectProject, createProject, fetchProjects } = useAppStore();
  const [newProjectName, setNewProjectName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim() || isCreating) return;
    
    setIsCreating(true);
    await createProject(newProjectName);
    setNewProjectName('');
    setIsCreating(false);
  };

  return (
    <div className="w-64 bg-gray-50 dark:bg-gray-800/30 border-r border-gray-200 dark:border-gray-700/50 flex flex-col p-3">
      <div className="flex items-center mb-4 px-2">
        <BrainCircuit className="h-6 w-6 mr-2 text-blue-500" />
        <h1 className="text-xl font-bold">AtaraxAI</h1>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto pr-1">
        {projects.map((project) => (
          <button
            key={project.project_id}
            onClick={() => selectProject(project.project_id)}
            className={`w-full flex items-center text-left p-2.5 rounded-lg transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
              selectedProjectId === project.project_id 
                ? 'bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-300 font-semibold' 
                : 'hover:bg-gray-200/70 dark:hover:bg-gray-700/50'
            }`}
          >
            <Folder size={16} className="mr-3 flex-shrink-0" />
            <span className="truncate">{project.name}</span>
          </button>
        ))}
      </div>
      <form onSubmit={handleCreateProject} className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700/50">
        <div className="relative">
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="New project..."
              disabled={isCreating}
              className="w-full pl-3 pr-10 py-2 rounded-lg border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition disabled:opacity-50"
            />
            <button 
              type="submit"
              disabled={!newProjectName.trim() || isCreating}
              className="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 text-gray-400 hover:text-blue-500 disabled:text-gray-300 dark:disabled:text-gray-500 rounded-md transition"
            >
              <Plus size={20} />
            </button>
        </div>
      </form>
    </div>
  );
};

export default ProjectSidebar;