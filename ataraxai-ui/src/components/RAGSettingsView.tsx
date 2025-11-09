// import React from 'react';
import { useAppStore } from '../store/AppContext';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Folder, Trash2, Plus, Database } from 'lucide-react';

export function RAGSettingsView() {
  const { ragSources, addRagSource, removeRagSource } = useAppStore();

  const handleAddDirectory = () => {
    // In a real app, this would open a directory picker
    const mockPath = `/Users/developer/New Directory ${Date.now()}`;
    addRagSource(mockPath);
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="flex-1 bg-background">
      {/* Header */}
      <div className="border-b border-border p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Database size={24} className="text-primary" />
            <h1>Knowledge Sources</h1>
          </div>
          <Button onClick={handleAddDirectory} className="bg-primary hover:bg-primary/90">
            <Plus size={16} className="mr-2" />
            Add Directory
          </Button>
        </div>
        <p className="text-muted-foreground mt-2">
          Manage directories and files that Atarax-AI can access for retrieval-augmented generation (RAG).
        </p>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-6">
          {ragSources.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Database size={48} className="text-muted-foreground mb-4" />
                <h3 className="mb-2">No knowledge sources configured</h3>
                <p className="text-muted-foreground text-center mb-6">
                  Add directories containing documents, code, or other files that you want Atarax-AI to reference.
                </p>
                <Button onClick={handleAddDirectory} className="bg-primary hover:bg-primary/90">
                  <Plus size={16} className="mr-2" />
                  Add Your First Directory
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {ragSources.map((source) => (
                <Card key={source.id} className="bg-card border-border">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className="flex-shrink-0">
                          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                            <Folder size={20} className="text-primary" />
                          </div>
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <p className="truncate font-medium">{source.path}</p>
                            <Badge variant="secondary" className="flex-shrink-0">
                              {source.type}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Added {formatDate(source.addedAt)}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 flex-shrink-0 ml-4">
                        <div className="text-sm text-muted-foreground">
                          Indexed • Active
                        </div>
                        
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                                onClick={() => removeRagSource(source.id)}
                              >
                                <Trash2 size={16} />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              Remove from knowledge sources
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {/* Information Card */}
          <Card className="mt-6 bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database size={20} className="text-primary" />
                How RAG Works
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm">
                <strong>Retrieval-Augmented Generation (RAG)</strong> allows Atarax-AI to access and reference 
                your local files and documents when answering questions.
              </p>
              <ul className="text-sm space-y-1 text-muted-foreground ml-4">
                <li>• Documents are automatically indexed and made searchable</li>
                <li>• Relevant content is retrieved based on your queries</li>
                <li>• AI responses include context from your knowledge sources</li>
                <li>• All processing happens locally for privacy and security</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  );
}