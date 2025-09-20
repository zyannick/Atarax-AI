import React, { useState } from 'react';
import { useAppStore } from '../store/AppContext';
import { AtaraxLogo } from './AtaraxLogo';
import { ProjectDialog } from './ProjectDialog';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './ui/dropdown-menu';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { 
  ChevronDown, 
  ChevronRight, 
  Folder, 
  MessageSquare, 
  Plus, 
  MoreHorizontal,
  Settings,
  Download,
  Database,
  ChevronLeft,
  Edit,
  Trash2,
  BarChart3
} from 'lucide-react';

export function LeftSidebar() {
  const {
    projects,
    sessions,
    selectedProjectId,
    selectedSessionId,
    currentView,
    sidebarCollapsed,
    selectProject,
    selectSession,
    addSession,
    getSessionsByProject,
    setCurrentView,
    toggleSidebar,
    renameProject,
    deleteProject,
  } = useAppStore();

  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set(['1']));
  const [renameProjectId, setRenameProjectId] = useState<string | null>(null);
  const [renameProjectName, setRenameProjectName] = useState('');
  const [deleteProjectId, setDeleteProjectId] = useState<string | null>(null);

  const toggleProject = (projectId: string) => {
    const newExpanded = new Set(expandedProjects);
    if (newExpanded.has(projectId)) {
      newExpanded.delete(projectId);
    } else {
      newExpanded.add(projectId);
    }
    setExpandedProjects(newExpanded);
  };

  const handleProjectSelect = (projectId: string) => {
    selectProject(projectId);
    setCurrentView('chat');
    if (!expandedProjects.has(projectId)) {
      toggleProject(projectId);
    }
  };

  const handleSessionSelect = (sessionId: string) => {
    selectSession(sessionId);
    setCurrentView('chat');
  };

  const handleNewChat = () => {
    if (selectedProjectId) {
      addSession(selectedProjectId);
    }
  };

  const handleRenameProject = (projectId: string, currentName: string) => {
    setRenameProjectId(projectId);
    setRenameProjectName(currentName);
  };

  const handleRenameSubmit = () => {
    if (renameProjectId && renameProjectName.trim()) {
      renameProject(renameProjectId, renameProjectName.trim());
      setRenameProjectId(null);
      setRenameProjectName('');
    }
  };

  const handleRenameCancel = () => {
    setRenameProjectId(null);
    setRenameProjectName('');
  };

  const handleDeleteProject = (projectId: string) => {
    setDeleteProjectId(projectId);
  };

  const handleDeleteSubmit = () => {
    if (deleteProjectId) {
      deleteProject(deleteProjectId);
      setDeleteProjectId(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteProjectId(null);
  };

  if (sidebarCollapsed) {
    return (
      <div className="w-16 bg-sidebar border-r border-sidebar-border flex flex-col items-center py-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="mb-4"
        >
          <AtaraxLogo size={20} />
        </Button>
        <div className="flex flex-col gap-2">
          <Button
            variant={currentView === 'chat' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('chat')}
          >
            <MessageSquare size={16} />
          </Button>
          <Button
            variant={currentView === 'rag-settings' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('rag-settings')}
          >
            <Database size={16} />
          </Button>
          <Button
            variant={currentView === 'model-manager' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('model-manager')}
          >
            <Download size={16} />
          </Button>
          <Button
            variant={currentView === 'benchmark' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('benchmark')}
          >
            <BarChart3 size={16} />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-sidebar border-r border-sidebar-border flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AtaraxLogo size={24} className="text-sidebar-primary" />
            <span className="font-medium text-sidebar-foreground">Atarax-AI</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
          >
            <ChevronLeft size={16} />
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <div className="p-4 border-b border-sidebar-border">
        <div className="grid grid-cols-2 gap-1">
          <Button
            variant={currentView === 'chat' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('chat')}
            className="flex-1"
          >
            <MessageSquare size={16} className="mr-2" />
            Chat
          </Button>
          <Button
            variant={currentView === 'rag-settings' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('rag-settings')}
            className="flex-1"
          >
            <Database size={16} className="mr-2" />
            RAG
          </Button>
          <Button
            variant={currentView === 'model-manager' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('model-manager')}
            className="flex-1"
          >
            <Download size={16} className="mr-2" />
            Models
          </Button>
          <Button
            variant={currentView === 'benchmark' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('benchmark')}
            className="flex-1"
          >
            <BarChart3 size={16} className="mr-2" />
            Benchmark
          </Button>
        </div>
      </div>

      {/* Projects and Sessions */}
      {currentView === 'chat' && (
        <>
          <div className="p-4 border-b border-sidebar-border space-y-2">
            <Button
              onClick={handleNewChat}
              className="w-full bg-sidebar-primary hover:bg-sidebar-primary/90"
              disabled={!selectedProjectId}
            >
              <Plus size={16} className="mr-2" />
              New Chat
            </Button>
            <ProjectDialog />
          </div>

          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {projects.map((project) => {
                const projectSessions = getSessionsByProject(project.id);
                const isExpanded = expandedProjects.has(project.id);
                const isSelected = selectedProjectId === project.id;

                return (
                  <Collapsible
                    key={project.id}
                    open={isExpanded}
                    onOpenChange={() => toggleProject(project.id)}
                  >
                    <div className="group relative">
                      <CollapsibleTrigger asChild>
                        <Button
                          variant="ghost"
                          className={`w-full justify-start p-2 h-auto ${
                            isSelected ? 'bg-sidebar-accent' : ''
                          }`}
                          onClick={() => handleProjectSelect(project.id)}
                        >
                          <div className="flex items-center gap-2 flex-1">
                            {isExpanded ? (
                              <ChevronDown size={16} />
                            ) : (
                              <ChevronRight size={16} />
                            )}
                            <Folder size={16} />
                            <span className="truncate">{project.name}</span>
                          </div>
                        </Button>
                      </CollapsibleTrigger>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute right-1 top-1 opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <MoreHorizontal size={14} />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                          <DropdownMenuItem 
                            onClick={() => handleRenameProject(project.id, project.name)}
                          >
                            <Edit size={14} className="mr-2" />
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            className="text-destructive"
                            onClick={() => handleDeleteProject(project.id)}
                          >
                            <Trash2 size={14} className="mr-2" />
                            Delete Project
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <CollapsibleContent className="ml-4">
                      {projectSessions.map((session) => (
                        <div key={session.id} className="group relative">
                          <Button
                            variant="ghost"
                            className={`w-full justify-start p-2 h-auto text-sm ${
                              selectedSessionId === session.id
                                ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                                : ''
                            }`}
                            onClick={() => handleSessionSelect(session.id)}
                          >
                            <div className="flex items-center gap-2 flex-1">
                              <MessageSquare size={14} />
                              <span className="truncate">{session.title}</span>
                            </div>
                          </Button>

                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="absolute right-1 top-1 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <MoreHorizontal size={12} />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent>
                              <DropdownMenuItem>
                                <Edit size={12} className="mr-2" />
                                Rename
                              </DropdownMenuItem>
                              <DropdownMenuItem className="text-destructive">
                                <Trash2 size={12} className="mr-2" />
                                Delete Session
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      ))}
                    </CollapsibleContent>
                  </Collapsible>
                );
              })}
            </div>
          </ScrollArea>
        </>
      )}

      {/* Rename Project Dialog */}
      <Dialog open={!!renameProjectId} onOpenChange={(open) => !open && handleRenameCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit size={20} className="text-primary" />
              Rename Project
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); handleRenameSubmit(); }} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="rename-project-name">Project Name</Label>
              <Input
                id="rename-project-name"
                placeholder="Enter new project name..."
                value={renameProjectName}
                onChange={(e) => setRenameProjectName(e.target.value)}
                autoFocus
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button 
                type="button" 
                variant="outline" 
                onClick={handleRenameCancel}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                disabled={!renameProjectName.trim()}
                className="bg-primary hover:bg-primary/90"
              >
                Rename
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      {/* Delete Project Dialog */}
      <Dialog open={!!deleteProjectId} onOpenChange={(open) => !open && handleDeleteCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trash2 size={20} className="text-primary" />
              Delete Project
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); handleDeleteSubmit(); }} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="delete-project-name">Project Name</Label>
              <Input
                id="delete-project-name"
                placeholder="Enter project name..."
                value={projects.find((project) => project.id === deleteProjectId)?.name || ''}
                readOnly
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button 
                type="button" 
                variant="outline" 
                onClick={handleDeleteCancel}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                className="bg-primary hover:bg-primary/90"
              >
                Delete
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}