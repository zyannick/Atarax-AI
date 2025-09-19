import React, { useState } from 'react';
import { useAppStore } from '../store/AppContext';
import { BenchmarkTest, BenchmarkSession, ModelInfo } from '../store/types';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { ScrollArea } from './ui/scroll-area';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import {
  Play,
  Pause,
  BarChart3,
  Trophy,
  Clock,
  Cpu,
  Zap,
  Brain,
  Plus,
  Settings,
  Download,
  CheckCircle,
  AlertCircle,
  TrendingUp,
} from 'lucide-react';

export function BenchmarkView() {
  const {
    models,
    benchmarkTests,
    benchmarkSessions,
    activeBenchmarkId,
    addBenchmarkSession,
    addBenchmarkTest,
    setActiveBenchmark,
  } = useAppStore();

  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedTests, setSelectedTests] = useState<string[]>([]);
  const [newTestDialog, setNewTestDialog] = useState(false);
  const [newTestName, setNewTestName] = useState('');
  const [newTestDescription, setNewTestDescription] = useState('');
  const [newTestCategory, setNewTestCategory] = useState<'performance' | 'accuracy' | 'reasoning' | 'memory'>('performance');
  const [newTestDuration, setNewTestDuration] = useState(60);

  const availableModels = models.filter(model => model.isDownloaded);
  const activeBenchmark = benchmarkSessions.find(session => session.id === activeBenchmarkId);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'performance': return <Zap className="h-4 w-4" />;
      case 'accuracy': return <Trophy className="h-4 w-4" />;
      case 'reasoning': return <Brain className="h-4 w-4" />;
      case 'memory': return <Cpu className="h-4 w-4" />;
      default: return <BarChart3 className="h-4 w-4" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'performance': return 'bg-yellow-500/20 text-yellow-300';
      case 'accuracy': return 'bg-green-500/20 text-green-300';
      case 'reasoning': return 'bg-blue-500/20 text-blue-300';
      case 'memory': return 'bg-purple-500/20 text-purple-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  const handleCreateBenchmark = () => {
    if (selectedModels.length > 0 && selectedTests.length > 0) {
      const sessionName = `Benchmark - ${new Date().toLocaleDateString()}`;
      addBenchmarkSession(sessionName, selectedModels, selectedTests);
      setSelectedModels([]);
      setSelectedTests([]);
    }
  };

  const handleCreateTest = () => {
    if (newTestName && newTestDescription) {
      addBenchmarkTest(newTestName, newTestDescription, newTestCategory, newTestDuration);
      setNewTestDialog(false);
      setNewTestName('');
      setNewTestDescription('');
      setNewTestCategory('performance');
      setNewTestDuration(60);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'running': return <Play className="h-4 w-4 text-blue-400 animate-pulse" />;
      case 'failed': return <AlertCircle className="h-4 w-4 text-red-400" />;
      default: return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return minutes > 0 ? `${minutes}m ${remainingSeconds}s` : `${remainingSeconds}s`;
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="flex items-center gap-2">
              <BarChart3 className="h-6 w-6 text-primary" />
              Model Benchmark Suite
            </h1>
            <p className="text-muted-foreground">
              Evaluate and compare AI model performance across various metrics
            </p>
          </div>
          <div className="flex gap-2">
            <Dialog open={newTestDialog} onOpenChange={setNewTestDialog}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  Create Test
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create Custom Benchmark Test</DialogTitle>
                  <DialogDescription>
                    Create a custom test to evaluate specific aspects of model performance.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="test-name">Test Name</Label>
                    <Input
                      id="test-name"
                      value={newTestName}
                      onChange={(e) => setNewTestName(e.target.value)}
                      placeholder="Enter test name..."
                    />
                  </div>
                  <div>
                    <Label htmlFor="test-description">Description</Label>
                    <Textarea
                      id="test-description"
                      value={newTestDescription}
                      onChange={(e) => setNewTestDescription(e.target.value)}
                      placeholder="Describe what this test evaluates..."
                    />
                  </div>
                  <div>
                    <Label htmlFor="test-category">Category</Label>
                    <Select value={newTestCategory} onValueChange={(value: any) => setNewTestCategory(value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="performance">Performance</SelectItem>
                        <SelectItem value="accuracy">Accuracy</SelectItem>
                        <SelectItem value="reasoning">Reasoning</SelectItem>
                        <SelectItem value="memory">Memory</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="test-duration">Estimated Duration (seconds)</Label>
                    <Input
                      id="test-duration"
                      type="number"
                      value={newTestDuration}
                      onChange={(e) => setNewTestDuration(parseInt(e.target.value) || 60)}
                      min="1"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setNewTestDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleCreateTest} disabled={!newTestName || !newTestDescription}>
                      Create Test
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            <Button 
              onClick={handleCreateBenchmark}
              disabled={selectedModels.length === 0 || selectedTests.length === 0}
              className="bg-primary hover:bg-primary/90"
            >
              <Play className="h-4 w-4 mr-2" />
              Run Benchmark
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6">
        <Tabs defaultValue="setup" className="h-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="setup">Setup</TabsTrigger>
            <TabsTrigger value="running">Running</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          </TabsList>

          <TabsContent value="setup" className="h-full">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
              {/* Model Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Download className="h-5 w-5" />
                    Select Models
                  </CardTitle>
                  <CardDescription>
                    Choose downloaded models to benchmark ({selectedModels.length} selected)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {availableModels.map((model) => (
                        <div key={model.id} className="flex items-center space-x-3 p-3 rounded-lg border border-border">
                          <Checkbox
                            id={`model-${model.id}`}
                            checked={selectedModels.includes(model.id)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedModels([...selectedModels, model.id]);
                              } else {
                                setSelectedModels(selectedModels.filter(id => id !== model.id));
                              }
                            }}
                          />
                          <div className="flex-1">
                            <label htmlFor={`model-${model.id}`} className="cursor-pointer">
                              <div className="font-medium">{model.name}</div>
                              <div className="text-sm text-muted-foreground">{model.size}</div>
                            </label>
                          </div>
                          <Badge variant="secondary">Ready</Badge>
                        </div>
                      ))}
                      {availableModels.length === 0 && (
                        <div className="text-center py-8 text-muted-foreground">
                          <Download className="h-8 w-8 mx-auto mb-2 opacity-50" />
                          <p>No downloaded models available</p>
                          <p className="text-sm">Download models from the Model Manager first</p>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>

              {/* Test Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Select Tests
                  </CardTitle>
                  <CardDescription>
                    Choose benchmark tests to run ({selectedTests.length} selected)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {benchmarkTests.map((test) => (
                        <div key={test.id} className="flex items-center space-x-3 p-3 rounded-lg border border-border">
                          <Checkbox
                            id={`test-${test.id}`}
                            checked={selectedTests.includes(test.id)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedTests([...selectedTests, test.id]);
                              } else {
                                setSelectedTests(selectedTests.filter(id => id !== test.id));
                              }
                            }}
                          />
                          <div className="flex-1">
                            <label htmlFor={`test-${test.id}`} className="cursor-pointer">
                              <div className="font-medium flex items-center gap-2">
                                {getCategoryIcon(test.category)}
                                {test.name}
                              </div>
                              <div className="text-sm text-muted-foreground">{test.description}</div>
                              <div className="text-xs text-muted-foreground mt-1">
                                <Clock className="h-3 w-3 inline mr-1" />
                                ~{formatDuration(test.estimatedDuration)}
                              </div>
                            </label>
                          </div>
                          <Badge className={getCategoryColor(test.category)}>
                            {test.category}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="running" className="h-full">
            <div className="space-y-6">
              {activeBenchmark ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {getStatusIcon(activeBenchmark.status)}
                      {activeBenchmark.name}
                    </CardTitle>
                    <CardDescription>
                      Testing {activeBenchmark.modelIds.length} models across {activeBenchmark.testIds.length} tests
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium">Overall Progress</span>
                          <span className="text-sm text-muted-foreground">{activeBenchmark.progress}%</span>
                        </div>
                        <Progress value={activeBenchmark.progress} className="h-2" />
                      </div>
                      
                      {activeBenchmark.status === 'running' && (
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            <Pause className="h-4 w-4 mr-2" />
                            Pause
                          </Button>
                          <Button variant="destructive" size="sm">
                            Stop
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <Play className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-lg font-medium">No Active Benchmark</p>
                    <p className="text-muted-foreground">Set up and run a benchmark to see progress here</p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="results" className="h-full">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Benchmark Results
                  </CardTitle>
                  <CardDescription>
                    Detailed performance metrics and comparisons
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {benchmarkSessions.length > 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <p>Results will appear here after completing benchmarks</p>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <BarChart3 className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No benchmark sessions yet</p>
                      <p className="text-sm">Run your first benchmark to see results</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="leaderboard" className="h-full">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="h-5 w-5" />
                    Model Leaderboard
                  </CardTitle>
                  <CardDescription>
                    Ranked model performance across all benchmarks
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-8 text-muted-foreground">
                    <Trophy className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Leaderboard will appear here</p>
                    <p className="text-sm">Complete benchmarks to see model rankings</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}