import { useEffect } from 'react';
import { ProjectSidebar } from './components/ProjectSidebar';
import { SessionSidebar } from './components/SessionSidebar';
import { ChatWindow } from './components/ChatView';
import { useAppStore } from './store';
import { useState } from 'react';
import { BrainCircuit, Folder, MessageSquare, Plus, Send, X, Settings, Sun, Moon } from 'lucide-react';

export const WorkspaceView = () => {
  const { error } = useAppStore();

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
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-sans">
      <ProjectSidebar />
      <SessionSidebar />
      <ChatWindow toggleTheme={toggleTheme} currentTheme={theme} />
    </div>
  );
};