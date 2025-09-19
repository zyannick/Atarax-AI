import React, { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store/AppContext';
import { AtaraxLogo } from './AtaraxLogo';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { 
  Send, 
  Mic, 
  Image, 
  Camera, 
  Paperclip, 
  Copy, 
  Settings,
  User
} from 'lucide-react';

export function ChatView() {
  const {
    selectedSessionId,
    sessions,
    messages,
    isTyping,
    getMessagesBySession,
    addMessage,
  } = useAppStore();

  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const currentSession = sessions.find(s => s.id === selectedSessionId);
  const currentMessages = selectedSessionId ? getMessagesBySession(selectedSessionId) : [];

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [currentMessages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !selectedSessionId) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    
    // Add user message
    addMessage(selectedSessionId, userMessage, 'user');

    // Simulate AI response (in real app, this would call the backend)
    setTimeout(() => {
      const responses = [
        "I understand you're asking about that. Let me help you with a detailed response.",
        "That's an interesting question. Based on my knowledge, here's what I can tell you...",
        "I can assist you with that. Here are some key points to consider:",
        "Let me break this down for you in a clear and organized way.",
      ];
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      addMessage(selectedSessionId, randomResponse, 'assistant');
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = (type: 'file' | 'image') => {
    if (type === 'file') {
      fileInputRef.current?.click();
    } else {
      imageInputRef.current?.click();
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  };

  if (!currentSession) {
    return (
      <div className="flex-1 flex items-center justify-center bg-background">
        <div className="text-center">
          <AtaraxLogo size={48} className="mx-auto mb-4 text-muted-foreground" />
          <h2>Welcome to Atarax-AI</h2>
          <p className="text-muted-foreground">Select a chat session to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <h1 className="truncate">{currentSession.title}</h1>
          <Button variant="ghost" size="sm">
            <Settings size={16} />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
        <div className="space-y-6 max-w-4xl mx-auto">
          {currentMessages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              {/* Avatar */}
              <Avatar className="w-8 h-8 flex-shrink-0">
                {message.role === 'user' ? (
                  <AvatarFallback className="bg-secondary">
                    <User size={16} />
                  </AvatarFallback>
                ) : (
                  <AvatarFallback className="bg-primary/10">
                    <AtaraxLogo size={16} className="text-primary" />
                  </AvatarFallback>
                )}
              </Avatar>

              {/* Message Content */}
              <div className={`flex-1 max-w-[70%] ${message.role === 'user' ? 'items-end' : 'items-start'} flex flex-col`}>
                <div
                  className={`rounded-lg px-4 py-3 group relative ${
                    message.role === 'user'
                      ? 'bg-secondary text-secondary-foreground'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  
                  {message.role === 'assistant' && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0"
                            onClick={() => copyToClipboard(message.content)}
                          >
                            <Copy size={12} />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          Copy message
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
                
                <span className="text-xs text-muted-foreground mt-1">
                  {formatTime(message.timestamp)}
                </span>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex gap-3">
              <Avatar className="w-8 h-8">
                <AvatarFallback className="bg-primary/10">
                  <AtaraxLogo size={16} className="text-primary" />
                </AvatarFallback>
              </Avatar>
              <div className="bg-muted rounded-lg px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t border-border p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-2">
            {/* Input Controls */}
            <div className="flex gap-1">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={isRecording ? 'text-destructive' : ''}
                      onClick={() => setIsRecording(!isRecording)}
                    >
                      <Mic size={16} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isRecording ? 'Stop recording' : 'Voice input'}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleFileUpload('image')}
                    >
                      <Image size={16} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Upload image
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <Camera size={16} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    Camera
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>

            {/* Text Input */}
            <div className="flex-1 relative">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Message Atarax-AI..."
                className="pr-20 min-h-[40px] bg-input-background border-border"
              />
              
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleFileUpload('file')}
                      >
                        <Paperclip size={16} />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Attach file
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <Button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim()}
                  className="bg-primary hover:bg-primary/90"
                >
                  <Send size={16} />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="*/*"
      />
      <input
        ref={imageInputRef}
        type="file"
        className="hidden"
        accept="image/*"
      />
    </div>
  );
}