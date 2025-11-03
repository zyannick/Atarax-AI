import React, { useState } from 'react';
import { useAppStore } from '../store/AppContext';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Plus, Folder } from 'lucide-react';

interface ProjectDialogProps {
  trigger?: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function ProjectDialog({ trigger, open, onOpenChange }: ProjectDialogProps) {
  const { addProject } = useAppStore();
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const dialogOpen = open !== undefined ? open : isDialogOpen;
  const setDialogOpen = onOpenChange || setIsDialogOpen;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (projectName.trim() && projectDescription.trim()) {
      addProject(projectName.trim(), projectDescription.trim());
      setProjectName('');
      setProjectDescription('');
      setDialogOpen(false);
    }
  };

  const handleCancel = () => {
    setProjectName('');
    setProjectDescription('');
    setDialogOpen(false);
  };

  const defaultTrigger = (
    <Button 
      variant="outline" 
      size="sm"
      className="w-full justify-start border-sidebar-border bg-sidebar hover:bg-sidebar-accent"
    >
      <Plus size={16} className="mr-2" />
      New Project
    </Button>
  );

  return (
    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Folder size={20} className="text-primary" />
            Create New Project
          </DialogTitle>
          <DialogDescription>
            Create a new project to organize your AI assistant conversations and settings.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="project-name">Project Name</Label>
            <Input
              id="project-name"
              placeholder="Enter project name..."
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              autoFocus
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="project-description">Description</Label>
            <Textarea
              id="project-description"
              placeholder="What is this project about... (e.g., 'RAG for financial reports')"
              value={projectDescription}
              onChange={(e) => setProjectDescription(e.target.value)}
              className="min-h-[80px]"
            />
          </div>
          
          <div className="flex justify-end gap-2">
            <Button 
              type="button" 
              variant="outline" 
              onClick={handleCancel}
            >
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={!projectName.trim() || !projectDescription.trim()}
              className="bg-primary hover:bg-primary/90"
            >
              Create Project
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
