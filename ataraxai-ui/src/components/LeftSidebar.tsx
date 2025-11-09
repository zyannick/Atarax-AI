import { useState, useEffect } from 'react';
import { useAppStore } from '../store/AppContext';
import { AtaraxLogo } from './AtaraxLogo';
import { ProjectDialog } from './ProjectDialog';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './ui/dropdown-menu';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import {
  ChevronDown,
  ChevronRight,
  Folder,
  MessageSquare,
  Plus,
  MoreHorizontal,
  Download,
  Database,
  ChevronLeft,
  Edit,
  Trash2,
  BarChart3,
  Lock,
  Loader2
} from 'lucide-react';

interface LeftSidebarProps {
  onLockVault?: () => void;
}

export function LeftSidebar({ onLockVault }: LeftSidebarProps) {
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
    fetchSessionsForProject,
    getSessionsByProject,
    setCurrentView,
    toggleSidebar,
    updateProject,
    deleteProject,
    renameSession,
    deleteSession,
  } = useAppStore();

  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());
  const [loadingProjects, setLoadingProjects] = useState<Set<string>>(new Set());
  const [updateProjectId, setUpdateProjectId] = useState<string | null>(null);
  const [updateProjectName, setUpdateProjectName] = useState('');
  const [updateProjectDescription, setUpdateProjectDescription] = useState('');
  const [deleteProjectId, setDeleteProjectId] = useState<string | null>(null);
  const [renameSessionId, setRenameSessionId] = useState<string | null>(null);
  const [renameSessionTitle, setRenameSessionTitle] = useState('');
  const [deleteSessionId, setDeleteSessionId] = useState<string | null>(null);

  useEffect(() => {
    if (selectedProjectId && !expandedProjects.has(selectedProjectId)) {
      const newExpanded = new Set(expandedProjects);
      newExpanded.add(selectedProjectId);
      setExpandedProjects(newExpanded);
      
      const projectSessions = getSessionsByProject(selectedProjectId);
      if (projectSessions.length === 0 && !loadingProjects.has(selectedProjectId)) {
        fetchSessionsForProject(selectedProjectId);
      }
    }
  }, [selectedProjectId, expandedProjects, getSessionsByProject, loadingProjects, fetchSessionsForProject]);

  const toggleProject = async (projectId: string) => {
    const newExpanded = new Set(expandedProjects);
    const wasExpanded = newExpanded.has(projectId);

    if (wasExpanded) {
      newExpanded.delete(projectId);
      setExpandedProjects(newExpanded);
    } else {
      newExpanded.add(projectId);
      setExpandedProjects(newExpanded);

      if (!loadingProjects.has(projectId)) {
        setLoadingProjects(new Set(loadingProjects).add(projectId));
        try {
          await fetchSessionsForProject(projectId);
        } catch (error) {
          console.error('Failed to fetch sessions:', error);
        } finally {
          setLoadingProjects(prev => {
            const next = new Set(prev);
            next.delete(projectId);
            return next;
          });
        }
      }
    }
  };

  const handleProjectSelect = async (projectId: string) => {
    selectProject(projectId);
    setCurrentView('chat');

    if (!expandedProjects.has(projectId) && !loadingProjects.has(projectId)) {
      setLoadingProjects(new Set(loadingProjects).add(projectId));
      try {
        await fetchSessionsForProject(projectId);
      } catch (error) {
        console.error('Failed to fetch sessions:', error);
      } finally {
        setLoadingProjects(prev => {
          const next = new Set(prev);
          next.delete(projectId);
          return next;
        });
      }
    }
  };

  const handleSessionSelect = (sessionId: string) => {
    selectSession(sessionId);
    setCurrentView('chat');
  };

  const handleNewChat = async () => {
    if (!selectedProjectId) {
      console.warn('No project selected, cannot create new chat');
      return;
    }

    try {
      console.log(`Creating new chat for project: ${selectedProjectId}`);
      await addSession(selectedProjectId);
      
      if (!expandedProjects.has(selectedProjectId)) {
        const newExpanded = new Set(expandedProjects);
        newExpanded.add(selectedProjectId);
        setExpandedProjects(newExpanded);
      }
      
      console.log('New chat created successfully');
    } catch (error) {
      console.error('Failed to create new chat:', error);
    }
  };

  const handleUpdateProject = (projectId: string, currentName: string, currentDescription: string) => {
    setUpdateProjectId(projectId);
    setUpdateProjectName(currentName);
    setUpdateProjectDescription(currentDescription || '');
  };

  const handleUpdateProjectSubmit = async () => {
    if (updateProjectId && updateProjectName.trim()) {
      try {
        await updateProject(updateProjectId, updateProjectName.trim(), updateProjectDescription.trim());
        setUpdateProjectId(null);
        setUpdateProjectName('');
        setUpdateProjectDescription('');
      } catch (error) {
        console.error('Failed to update project:', error);
      }
    }
  };

  const handleUpdateProjectCancel = () => {
    setUpdateProjectId(null);
    setUpdateProjectName('');
    setUpdateProjectDescription('');
  };

  const handleDeleteProject = (projectId: string) => {
    setDeleteProjectId(projectId);
  };

  const handleDeleteProjectSubmit = async () => {
    if (deleteProjectId) {
      try {
        await deleteProject(deleteProjectId);
        setDeleteProjectId(null);
        
        if (expandedProjects.has(deleteProjectId)) {
          const newExpanded = new Set(expandedProjects);
          newExpanded.delete(deleteProjectId);
          setExpandedProjects(newExpanded);
        }
      } catch (error) {
        console.error('Failed to delete project:', error);
      }
    }
  };

  const handleDeleteProjectCancel = () => {
    setDeleteProjectId(null);
  };

  const handleRenameSession = (sessionId: string, currentTitle: string) => {
    setRenameSessionId(sessionId);
    setRenameSessionTitle(currentTitle);
  };

  const handleRenameSessionSubmit = async () => {
    if (renameSessionId && renameSessionTitle.trim()) {
      try {
        await renameSession(renameSessionId, renameSessionTitle.trim());
        setRenameSessionId(null);
        setRenameSessionTitle('');
      } catch (error) {
        console.error('Failed to rename session:', error);
      }
    }
  };

  const handleRenameSessionCancel = () => {
    setRenameSessionId(null);
    setRenameSessionTitle('');
  };

  const handleDeleteSession = (sessionId: string) => {
    setDeleteSessionId(sessionId);
  };

  const handleDeleteSessionSubmit = async () => {
    if (deleteSessionId) {
      try {
        await deleteSession(deleteSessionId);
        setDeleteSessionId(null);
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const handleDeleteSessionCancel = () => {
    setDeleteSessionId(null);
  };

  if (sidebarCollapsed) {
    return (
      <div className="w-16 bg-sidebar border-r border-sidebar-border flex flex-col items-center py-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="mb-4"
          aria-label="Expand sidebar"
        >
          <AtaraxLogo className="w-5 h-5" />
        </Button>
        <div className="flex flex-col gap-2">
          <Button
            variant={currentView === 'chat' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('chat')}
            aria-label="Chat view"
          >
            <MessageSquare size={16} />
          </Button>
          <Button
            variant={currentView === 'rag-settings' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('rag-settings')}
            aria-label="RAG settings"
          >
            <Database size={16} />
          </Button>
          <Button
            variant={currentView === 'model-manager' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('model-manager')}
            aria-label="Model manager"
          >
            <Download size={16} />
          </Button>
          <Button
            variant={currentView === 'benchmark' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setCurrentView('benchmark')}
            aria-label="Benchmark"
          >
            <BarChart3 size={16} />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-sidebar border-r border-sidebar-border flex flex-col h-screen">
      <div className="p-4 border-b border-sidebar-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AtaraxLogo className="w-6 h-6 text-sidebar-primary" />
            <span className="font-medium text-sidebar-foreground">Atarax-AI</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
            aria-label="Collapse sidebar"
          >
            <ChevronLeft size={16} />
          </Button>
        </div>
      </div>

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
              {projects.length === 0 ? (
                <div className="text-center text-muted-foreground text-sm py-8">
                  <p>No projects yet</p>
                  <p className="text-xs mt-2">Create a project to get started</p>
                </div>
              ) : (
                projects.map((project) => {
                  const projectSessions = getSessionsByProject(project.id);
                  const isExpanded = expandedProjects.has(project.id);
                  const isSelected = selectedProjectId === project.id;
                  const isLoading = loadingProjects.has(project.id);

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
                            onClick={(e) => {
                              e.preventDefault();
                              handleProjectSelect(project.id);
                            }}
                          >
                            <div className="flex items-center gap-2 flex-1">
                              {isLoading ? (
                                <Loader2 size={16} className="animate-spin" />
                              ) : isExpanded ? (
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
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreHorizontal size={14} />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuItem
                              onClick={() => handleUpdateProject(project.id, project.name, project.description)}
                            >
                              <Edit size={14} className="mr-2" />
                              Update
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
                        {projectSessions.length === 0 && !isLoading && (
                          <div className="text-xs text-muted-foreground p-2 italic">
                            No sessions yet
                          </div>
                        )}
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
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <MoreHorizontal size={12} />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent>
                                <DropdownMenuItem
                                  onClick={() => handleRenameSession(session.id, session.title)}
                                >
                                  <Edit size={12} className="mr-2" />
                                  Rename
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  className="text-destructive"
                                  onClick={() => handleDeleteSession(session.id)}
                                >
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
                })
              )}
            </div>
          </ScrollArea>
        </>
      )}

      {currentView !== 'chat' && (
        <div className="flex-1" />
      )}

      {onLockVault && (
        <div className="p-4 border-t border-sidebar-border">
          <Button
            variant="outline"
            size="sm"
            onClick={onLockVault}
            className="w-full border-border hover:bg-accent"
          >
            <Lock size={16} className="mr-2" />
            Lock Vault
          </Button>
        </div>
      )}

      <Dialog open={!!updateProjectId} onOpenChange={(open : boolean) => !open && handleUpdateProjectCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit size={20} className="text-primary" />
              Update Project
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); handleUpdateProjectSubmit(); }} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="update-project-name">Project Name</Label>
              <Input
                id="update-project-name"
                placeholder="Enter new project name..."
                value={updateProjectName}
                onChange={(e) => setUpdateProjectName(e.target.value)}
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="update-project-description">Description</Label>
              <Textarea
                id="update-project-description"
                placeholder="Enter project description..."
                value={updateProjectDescription}
                onChange={(e) => setUpdateProjectDescription(e.target.value)}
                rows={3}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleUpdateProjectCancel}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!updateProjectName.trim()}
                className="bg-primary hover:bg-primary/90"
              >
                Update
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={!!deleteProjectId} onOpenChange={(open : boolean) => !open && handleDeleteProjectCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trash2 size={20} className="text-destructive" />
              Delete Project
            </DialogTitle>
            <DialogDescription>
              This action cannot be undone. All sessions in this project will be deleted.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Project Name</Label>
              <Input
                value={projects.find((p) => p.id === deleteProjectId)?.name || ''}
                readOnly
                disabled
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleDeleteProjectCancel}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDeleteProjectSubmit}
                variant="destructive"
              >
                Delete
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={!!renameSessionId} onOpenChange={(open : boolean) => !open && handleRenameSessionCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit size={20} className="text-primary" />
              Rename Session
            </DialogTitle>
          </DialogHeader>
          <form onSubmit={(e) => { e.preventDefault(); handleRenameSessionSubmit(); }} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="rename-session-title">Session Title</Label>
              <Input
                id="rename-session-title"
                placeholder="Enter new session title..."
                value={renameSessionTitle}
                onChange={(e) => setRenameSessionTitle(e.target.value)}
                autoFocus
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleRenameSessionCancel}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!renameSessionTitle.trim()}
                className="bg-primary hover:bg-primary/90"
              >
                Rename
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={!!deleteSessionId} onOpenChange={(open : boolean) => !open && handleDeleteSessionCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Trash2 size={20} className="text-destructive" />
              Delete Session
            </DialogTitle>
            <DialogDescription>
              This action cannot be undone. This session and all its messages will be deleted.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Session Title</Label>
              <Input
                value={sessions.find((s) => s.id === deleteSessionId)?.title || ''}
                readOnly
                disabled
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleDeleteSessionCancel}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDeleteSessionSubmit}
                variant="destructive"
              >
                Delete
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}