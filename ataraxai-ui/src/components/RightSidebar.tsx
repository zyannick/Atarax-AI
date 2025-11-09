// import React from 'react';
import { useAppStore } from '../store/AppContext';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { 
  FileText, 
  Clock, 
  Settings, 
  Zap,
  Database,
  Cpu,
  Info
} from 'lucide-react';

export function RightSidebar() {
  const { currentView, selectedSessionId, ragSources, models } = useAppStore();

  if (currentView === 'rag-settings') {
    return (
      <div className="w-80 bg-sidebar border-l border-sidebar-border flex flex-col">
        <div className="p-4 border-b border-sidebar-border">
          <h3 className="flex items-center gap-2">
            <Info size={18} />
            RAG Configuration
          </h3>
        </div>
        
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Indexing Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Total Sources</span>
                  <Badge variant="secondary">{ragSources.length}</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Indexed Files</span>
                  <Badge variant="secondary">2,847</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Vector Embeddings</span>
                  <Badge variant="secondary">45,623</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Recent Activity</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <Clock size={12} className="text-muted-foreground" />
                    <span>Indexed 47 new files</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock size={12} className="text-muted-foreground" />
                    <span>Updated embeddings</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </div>
    );
  }

  if (currentView === 'model-manager') {
    return (
      <div className="w-80 bg-sidebar border-l border-sidebar-border flex flex-col">
        <div className="p-4 border-b border-sidebar-border">
          <h3 className="flex items-center gap-2">
            <Cpu size={18} />
            System Resources
          </h3>
        </div>
        
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Storage Usage</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Models Downloaded</span>
                  <Badge variant="secondary">{models.filter(m => m.isDownloaded).length}</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Total Size</span>
                  <Badge variant="secondary">
                    {models.reduce((total, model) => {
                      if (model.isDownloaded) {
                        const size = parseFloat(model.size);
                        return total + (isNaN(size) ? 0 : size);
                      }
                      return total;
                    }, 0).toFixed(1)} GB
                  </Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Available Space</span>
                  <Badge variant="outline">847 GB</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>CPU Usage</span>
                  <Badge variant="secondary">23%</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Memory</span>
                  <Badge variant="secondary">4.2/16 GB</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>GPU</span>
                  <Badge variant="secondary">Available</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </div>
    );
  }

  // Chat view context sidebar
  if (currentView === 'chat' && selectedSessionId) {
    return (
      <div className="w-80 bg-sidebar border-l border-sidebar-border flex flex-col">
        <div className="p-4 border-b border-sidebar-border">
          <h3 className="flex items-center gap-2">
            <Database size={18} />
            Context Sources
          </h3>
        </div>
        
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <FileText size={16} />
                  Referenced Documents
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-xs space-y-2">
                  <div className="p-2 bg-muted rounded border">
                    <div className="font-medium">react-patterns.md</div>
                    <div className="text-muted-foreground">~/Documents/Projects/</div>
                    <Badge variant="outline" className="text-xs mt-1">
                      Relevance: 94%
                    </Badge>
                  </div>
                  <div className="p-2 bg-muted rounded border">
                    <div className="font-medium">api-documentation.md</div>
                    <div className="text-muted-foreground">~/Knowledge Base/</div>
                    <Badge variant="outline" className="text-xs mt-1">
                      Relevance: 87%
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Session Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Model</span>
                  <Badge variant="secondary">llama2-7b</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Temperature</span>
                  <Badge variant="outline">0.7</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Max Tokens</span>
                  <Badge variant="outline">2048</Badge>
                </div>
                <Separator />
                <Button variant="outline" size="sm" className="w-full">
                  <Settings size={14} className="mr-2" />
                  Adjust Settings
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap size={16} />
                  Quick Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" size="sm" className="w-full justify-start">
                  Export Chat
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start">
                  Clear Context
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start">
                  Session Analytics
                </Button>
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      </div>
    );
  }

  return null;
}