export interface Project {
  id: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatSession {
  id: string;
  projectId: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Message {
  id: string;
  sessionId: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  type: 'text' | 'image' | 'voice' | 'video';
  metadata?: {
    fileUrl?: string;
    fileName?: string;
    fileSize?: number;
  };
}

export interface RagSource {
  id: string;
  path: string;
  type: 'directory' | 'file';
  addedAt: Date;
}

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  downloadProgress?: number;
  isDownloading?: boolean;
  isDownloaded: boolean;
}

export interface BenchmarkTest {
  id: string;
  name: string;
  description: string;
  category: 'performance' | 'accuracy' | 'reasoning' | 'memory';
  estimatedDuration: number; // in seconds
}

export interface BenchmarkResult {
  id: string;
  modelId: string;
  testId: string;
  score: number;
  maxScore: number;
  duration: number; // in milliseconds
  tokensPerSecond?: number;
  memoryUsage?: number; // in MB
  completedAt: Date;
  details?: any;
}

export interface BenchmarkSession {
  id: string;
  name: string;
  modelIds: string[];
  testIds: string[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  startedAt?: Date;
  completedAt?: Date;
  results: BenchmarkResult[];
}

export type AppView = 'chat' | 'rag-settings' | 'model-manager' | 'benchmark';
export type AppStatus = 'locked' | 'unlocked' | 'loading';