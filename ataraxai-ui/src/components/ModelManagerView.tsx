import React, { useState } from 'react';
import { useAppStore } from '../store/AppContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { 
  Download, 
  Search, 
  CheckCircle, 
  Clock, 
  HardDrive,
  Trash2,
  ExternalLink
} from 'lucide-react';

export function ModelManagerView() {
  const { 
    models, 
    searchQuery, 
    setSearchQuery, 
    addModel, 
    updateModelProgress, 
    markModelDownloaded 
  } = useAppStore();

  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    
    // Simulate search delay
    setTimeout(() => {
      setIsSearching(false);
    }, 1500);
  };

  const handleDownloadModel = (modelName: string) => {
    // In a real app, this would initiate the actual download
    const size = `${(Math.random() * 5 + 1).toFixed(1)} GB`;
    addModel(modelName, size);
    
    // Simulate download progress
    const modelId = Date.now().toString();
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress >= 100) {
        progress = 100;
        markModelDownloaded(modelId);
        clearInterval(interval);
      } else {
        updateModelProgress(modelId, Math.round(progress));
      }
    }, 500);
  };

  const formatFileSize = (size: string) => {
    return size;
  };

  const getModelStatus = (model: typeof models[0]) => {
    if (model.isDownloaded) {
      return { icon: CheckCircle, text: 'Downloaded', color: 'text-green-500' };
    } else if (model.isDownloading) {
      return { icon: Clock, text: 'Downloading', color: 'text-blue-500' };
    }
    return { icon: Download, text: 'Available', color: 'text-muted-foreground' };
  };

  const mockSearchResults = [
    { name: 'llama2-13b-chat', size: '7.3 GB', description: 'Large language model for conversational AI' },
    { name: 'codellama-7b-instruct', size: '3.8 GB', description: 'Code-focused language model' },
    { name: 'mistral-7b-openorca', size: '4.1 GB', description: 'Fine-tuned Mistral model' },
    { name: 'neural-chat-7b', size: '4.0 GB', description: 'Optimized for chat applications' },
  ];

  return (
    <div className="flex-1 bg-background">
      {/* Header */}
      <div className="border-b border-border p-6">
        <div className="flex items-center gap-3 mb-4">
          <Download size={24} className="text-primary" />
          <h1>Model Manager</h1>
        </div>
        <p className="text-muted-foreground">
          Download and manage AI models for local inference. All models run entirely on your device.
        </p>
      </div>

      <div className="flex-1 flex flex-col">
        {/* Search Section */}
        <div className="border-b border-border p-6">
          <h2 className="mb-4">Discover Models</h2>
          <div className="flex gap-2 max-w-2xl">
            <div className="flex-1 relative">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search Hugging Face models..."
                className="pl-10 bg-input-background"
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>
            <Button 
              onClick={handleSearch}
              disabled={isSearching || !searchQuery.trim()}
              className="bg-primary hover:bg-primary/90"
            >
              {isSearching ? (
                <>
                  <Clock size={16} className="mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search size={16} className="mr-2" />
                  Search
                </>
              )}
            </Button>
          </div>

          {/* Search Results */}
          {searchQuery && (
            <div className="mt-4 space-y-3">
              <h3 className="text-sm font-medium text-muted-foreground">Search Results</h3>
              <div className="grid gap-3">
                {mockSearchResults
                  .filter(model => model.name.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((model, index) => (
                  <Card key={index} className="bg-card border-border">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-medium">{model.name}</h4>
                            <Badge variant="outline">{model.size}</Badge>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                                    <ExternalLink size={12} />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  View on Hugging Face
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                          <p className="text-sm text-muted-foreground">{model.description}</p>
                        </div>
                        <Button
                          onClick={() => handleDownloadModel(model.name)}
                          className="ml-4"
                          disabled={models.some(m => m.name === model.name)}
                        >
                          <Download size={16} className="mr-2" />
                          Download
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Downloaded Models */}
        <ScrollArea className="flex-1">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2>Downloaded Models</h2>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <HardDrive size={16} />
                {models.filter(m => m.isDownloaded).length} models • {' '}
                {models.reduce((total, model) => {
                  if (model.isDownloaded) {
                    const size = parseFloat(model.size);
                    return total + (isNaN(size) ? 0 : size);
                  }
                  return total;
                }, 0).toFixed(1)} GB used
              </div>
            </div>

            {models.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Download size={48} className="text-muted-foreground mb-4" />
                  <h3 className="mb-2">No models downloaded</h3>
                  <p className="text-muted-foreground text-center mb-6">
                    Search and download AI models to start using Atarax-AI for local inference.
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {models.map((model) => {
                  const status = getModelStatus(model);
                  const StatusIcon = status.icon;

                  return (
                    <Card key={model.id} className="bg-card border-border">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3 flex-1">
                            <div className={`w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center`}>
                              <StatusIcon size={20} className={status.color} />
                            </div>
                            
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <h4 className="font-medium truncate">{model.name}</h4>
                                <Badge variant="outline">{model.size}</Badge>
                                <Badge 
                                  variant={model.isDownloaded ? 'default' : 'secondary'}
                                  className={model.isDownloaded ? 'bg-green-500/10 text-green-500' : ''}
                                >
                                  {status.text}
                                </Badge>
                              </div>
                              
                              {model.isDownloading && typeof model.downloadProgress === 'number' && (
                                <div className="flex items-center gap-2">
                                  <Progress value={model.downloadProgress} className="flex-1" />
                                  <span className="text-sm text-muted-foreground min-w-0">
                                    {model.downloadProgress}%
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center gap-2 ml-4">
                            {model.isDownloaded && (
                              <Button variant="outline" size="sm">
                                Use Model
                              </Button>
                            )}
                            
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                                  >
                                    <Trash2 size={16} />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  Delete model
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}