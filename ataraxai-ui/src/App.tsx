import React, { useState, useEffect, useRef } from 'react';
import { BrainCircuit, Folder, MessageSquare, Plus, Send, X, Settings, Sun, Moon, User, Sparkles } from 'lucide-react';

const useAppStore = () => {
  const [projects, setProjects] = useState([
    { project_id: '1', name: 'AI Research', description: 'Research project' },
    { project_id: '2', name: 'Web Development', description: 'Web dev project' }
  ]);
  const [selectedProjectId, setSelectedProjectId] = useState('1');
  const [sessions, setSessions] = useState([
    { session_id: '1', title: 'Getting Started', project_id: '1' },
    { session_id: '2', title: 'Advanced Topics', project_id: '1' }
  ]);
  const [selectedSessionId, setSelectedSessionId] = useState('1');
  const [messages, setMessages] = useState([
    { id: '1', role: 'assistant', content: 'Hello! How can I help you today?' },
    { id: '2', role: 'user', content: 'Can you explain machine learning?' },
    { id: '3', role: 'assistant', content: 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.' }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const createProject = async (name) => {
    const newProject = { project_id: Date.now().toString(), name, description: 'New project' };
    setProjects(prev => [...prev, newProject]);
    setSelectedProjectId(newProject.project_id);
  };

  const selectProject = (projectId) => {
    setSelectedProjectId(projectId);
    setSelectedSessionId(null);
    setMessages([]);
  };

  const createSession = async (projectId, title) => {
    const newSession = { session_id: Date.now().toString(), title, project_id: projectId };
    setSessions(prev => [...prev, newSession]);
    setSelectedSessionId(newSession.session_id);
  };

  const selectSession = (sessionId) => {
    setSelectedSessionId(sessionId);
    if (sessionId === '1') {
      setMessages([
        { id: '1', role: 'assistant', content: 'Hello! How can I help you today?' },
        { id: '2', role: 'user', content: 'Can you explain machine learning?' },
        { id: '3', role: 'assistant', content: 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.' }
      ]);
    } else {
      setMessages([{ id: '1', role: 'assistant', content: 'Welcome to a new session!' }]);
    }
  };

  const sendMessage = async (sessionId, query) => {
    const userMessage = { id: Date.now().toString(), role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);
    setTimeout(() => {
      const assistantMessage = { 
        id: (Date.now() + 1).toString(), 
        role: 'assistant', 
        content: `I received your message: "${query}". This is a simulated response from the AI assistant.` 
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1000);
  };

  return {
    projects,
    selectedProjectId,
    sessions: sessions.filter(s => s.project_id === selectedProjectId),
    selectedSessionId,
    messages,
    isLoading,
    createProject,
    selectProject,
    createSession,
    selectSession,
    sendMessage
  };
};

const ProjectSidebar = () => {
  const { projects, selectedProjectId, selectProject, createProject } = useAppStore();
  const [newProjectName, setNewProjectName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateProject = async () => {
    if (!newProjectName.trim() || isCreating) return;
    setIsCreating(true);
    await createProject(newProjectName);
    setNewProjectName('');
    setIsCreating(false);
  };

  return (
    <div className="w-72 bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 backdrop-blur-xl border-r border-blue-500/20 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-blue-500/20">
        <div className="flex items-center mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
            <BrainCircuit className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">AtaraxAI</h1>
            <p className="text-blue-300 text-sm opacity-80">Projects</p>
          </div>
        </div>
      </div>

      {/* Projects List */}
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-2">
          {projects.map((project) => (
            <button
              key={project.project_id}
              onClick={() => selectProject(project.project_id)}
              className={`w-full group flex items-center text-left p-4 rounded-xl transition-all duration-300 ${
                selectedProjectId === project.project_id
                  ? 'bg-gradient-to-r from-blue-500/30 to-blue-600/20 border border-blue-400/30 shadow-lg shadow-blue-500/10'
                  : 'hover:bg-white/5 border border-transparent hover:border-blue-500/20'
              }`}
            >
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center mr-3 transition-all duration-300 ${
                selectedProjectId === project.project_id
                  ? 'bg-blue-500/30 shadow-lg shadow-blue-500/20'
                  : 'bg-white/10 group-hover:bg-blue-500/20'
              }`}>
                <Folder size={18} className={`${
                  selectedProjectId === project.project_id ? 'text-blue-300' : 'text-blue-400'
                }`} />
              </div>
              <div className="flex-1">
                <span className={`font-medium block truncate ${
                  selectedProjectId === project.project_id ? 'text-white' : 'text-blue-100'
                }`}>
                  {project.name}
                </span>
                <span className="text-blue-300 text-sm opacity-60">
                  {project.description}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Create Project */}
      <div className="p-4 border-t border-blue-500/20">
        <div className="relative">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleCreateProject();
              }
            }}
            placeholder="Create new project..."
            disabled={isCreating}
            className="w-full pl-4 pr-12 py-3 rounded-xl bg-white/10 border border-blue-500/20 text-white placeholder-blue-300/60 focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-blue-400/50 transition-all duration-300 backdrop-blur-sm"
          />
          <button
            type="button"
            onClick={handleCreateProject}
            disabled={!newProjectName.trim() || isCreating}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

const SessionSidebar = () => {
  const {
    sessions,
    selectedSessionId,
    selectSession,
    createSession,
    selectedProjectId
  } = useAppStore();
  const [newSessionTitle, setNewSessionTitle] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreateSession = async () => {
    if (!newSessionTitle.trim() || !selectedProjectId || isCreating) return;
    setIsCreating(true);
    await createSession(selectedProjectId, newSessionTitle);
    setNewSessionTitle('');
    setIsCreating(false);
  };

  if (!selectedProjectId) {
    return (
      <div className="w-80 bg-gradient-to-br from-slate-800 to-slate-900 border-r border-blue-500/20 flex flex-col items-center justify-center text-center p-8">
        <div className="w-16 h-16 bg-gradient-to-br from-blue-400/20 to-blue-600/20 rounded-2xl flex items-center justify-center mb-4">
          <Folder size={32} className="text-blue-400" />
        </div>
        <h3 className="font-semibold text-xl text-white mb-2">No Project Selected</h3>
        <p className="text-blue-300/80">Select a project to view its sessions</p>
      </div>
    );
  }

  return (
    <div className="w-80 bg-gradient-to-br from-slate-800 to-slate-900 border-r border-blue-500/20 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-blue-500/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Sessions</h2>
            <p className="text-blue-300 text-sm opacity-80">{sessions.length} active sessions</p>
          </div>
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl flex items-center justify-center">
            <MessageSquare size={18} className="text-blue-400" />
          </div>
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-2">
          {sessions.map((session) => (
            <button
              key={session.session_id}
              onClick={() => selectSession(session.session_id)}
              className={`w-full group text-left p-4 rounded-xl transition-all duration-300 ${
                selectedSessionId === session.session_id
                  ? 'bg-gradient-to-r from-blue-500/20 to-blue-600/10 border border-blue-400/30 shadow-lg shadow-blue-500/10'
                  : 'hover:bg-white/5 border border-transparent hover:border-blue-500/20'
              }`}
            >
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-3 transition-all duration-300 ${
                  selectedSessionId === session.session_id ? 'bg-blue-400' : 'bg-blue-500/50'
                }`}></div>
                <span className={`font-medium truncate ${
                  selectedSessionId === session.session_id ? 'text-white' : 'text-blue-100'
                }`}>
                  {session.title}
                </span>
              </div>
            </button>
          ))}
          {sessions.length === 0 && (
            <div className="text-center py-8">
              <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center mx-auto mb-3">
                <MessageSquare size={20} className="text-blue-400" />
              </div>
              <p className="text-blue-300/80 text-sm">No sessions yet</p>
              <p className="text-blue-400/60 text-xs mt-1">Create your first session below</p>
            </div>
          )}
        </div>
      </div>

      {/* Create Session */}
      <div className="p-4 border-t border-blue-500/20">
        <div className="relative">
          <input
            type="text"
            value={newSessionTitle}
            onChange={(e) => setNewSessionTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleCreateSession();
              }
            }}
            placeholder="Create new session..."
            disabled={isCreating}
            className="w-full pl-4 pr-12 py-3 rounded-xl bg-white/10 border border-blue-500/20 text-white placeholder-blue-300/60 focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-blue-400/50 transition-all duration-300 backdrop-blur-sm"
          />
          <button
            type="button"
            onClick={handleCreateSession}
            disabled={!newSessionTitle.trim() || isCreating}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

const ChatView = ({ toggleTheme, currentTheme }) => {
  const { messages, sendMessage, selectedSessionId, isLoading } = useAppStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !selectedSessionId) return;
    
    await sendMessage(selectedSessionId, input);
    setInput('');
  };

  const autoResize = (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = e.target.scrollHeight + 'px';
  };

  if (!selectedSessionId) {
    return (
      <div className="flex-1 bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex flex-col items-center justify-center text-center p-8">
        <div className="w-20 h-20 bg-gradient-to-br from-blue-400/20 to-blue-600/20 rounded-3xl flex items-center justify-center mb-6">
          <MessageSquare size={40} className="text-blue-500" />
        </div>
        <h3 className="font-bold text-2xl text-slate-700 mb-2">Ready to Chat</h3>
        <p className="text-slate-500 max-w-md">Select a session from the sidebar to start your conversation with AtaraxAI</p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex flex-col">
      {/* Header */}
      <div className="bg-white/70 backdrop-blur-xl border-b border-blue-200/50 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
              <Sparkles size={20} className="text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-800">AtaraxAI Chat</h2>
              <p className="text-blue-600 text-sm">AI Assistant Ready</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={toggleTheme} 
              className="p-3 rounded-xl bg-white/70 hover:bg-white/90 border border-blue-200/50 transition-all duration-300 shadow-sm hover:shadow-md"
              title={`Switch to ${currentTheme === 'light' ? 'dark' : 'light'} theme`}
            >
              {currentTheme === 'light' ? <Moon size={18} className="text-blue-600" /> : <Sun size={18} className="text-blue-600" />}
            </button>
            <button 
              className="p-3 rounded-xl bg-white/70 hover:bg-white/90 border border-blue-200/50 transition-all duration-300 shadow-sm hover:shadow-md"
              title="Settings"
            >
              <Settings size={18} className="text-blue-600" />
            </button>
          </div>
        </div>
      </div>

      {/* Message List */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' ? (
                <>
                  <div className="max-w-2xl px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-3xl rounded-br-lg shadow-lg">
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  </div>
                  <div className="w-10 h-10 bg-gradient-to-br from-slate-400 to-slate-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
                    <User size={18} className="text-white" />
                  </div>
                </>
              ) : (
                <>
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
                    <BrainCircuit size={18} className="text-white" />
                  </div>
                  <div className={`max-w-2xl px-6 py-4 rounded-3xl rounded-bl-lg shadow-lg ${
                    msg.role === 'error'
                      ? 'bg-gradient-to-r from-red-50 to-red-100 border border-red-200 text-red-700'
                      : 'bg-white/90 backdrop-blur-sm border border-blue-200/50 text-slate-700'
                  }`}>
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  </div>
                </>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mr-4 shadow-lg">
                <BrainCircuit size={18} className="text-white" />
              </div>
              <div className="bg-white/90 backdrop-blur-sm border border-blue-200/50 rounded-3xl rounded-bl-lg px-6 py-4 shadow-lg">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Message Input */}
      <div className="p-6 bg-white/70 backdrop-blur-xl border-t border-blue-200/50">
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            <textarea
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoResize(e);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              rows={1}
              disabled={isLoading}
              className="w-full pl-6 pr-16 py-4 rounded-2xl bg-white/90 backdrop-blur-sm border border-blue-200/50 text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-blue-400/50 transition-all duration-300 resize-none min-h-[56px] max-h-[120px] shadow-lg"
              style={{ height: 'auto' }}
            />
            <button 
              type="button"
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-gray-800 font-sans">
      <ProjectSidebar />
      <SessionSidebar />
      <ChatView toggleTheme={toggleTheme} currentTheme={theme} />
    </div>
  );
}