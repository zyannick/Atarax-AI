import { StateCreator } from 'zustand';
import { Message } from './types';

export interface MessageSlice {
  messages: Message[];
  isTyping: boolean;
  
  // Actions
  addMessage: (sessionId: string, content: string, role: 'user' | 'assistant', type?: 'text' | 'image' | 'voice' | 'video', metadata?: any) => void;
  getMessagesBySession: (sessionId: string) => Message[];
  setTyping: (typing: boolean) => void;
  clearMessages: (sessionId: string) => void;
}

export const createMessageSlice: StateCreator<MessageSlice> = (set, get) => ({
  messages: [
    {
      id: '1',
      sessionId: '1',
      content: 'Hello! I\'m Atarax-AI, your local AI assistant. How can I help you today?',
      role: 'assistant',
      timestamp: new Date(),
      type: 'text',
    },
  ],
  isTyping: false,

  addMessage: (sessionId: string, content: string, role: 'user' | 'assistant', type = 'text', metadata) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      sessionId,
      content,
      role,
      timestamp: new Date(),
      type,
      metadata,
    };
    
    set((state) => ({
      messages: [...state.messages, newMessage],
    }));
  },

  getMessagesBySession: (sessionId: string) => {
    return get().messages.filter((message) => message.sessionId === sessionId);
  },

  setTyping: (typing: boolean) => {
    set({ isTyping: typing });
  },

  clearMessages: (sessionId: string) => {
    set((state) => ({
      messages: state.messages.filter((message) => message.sessionId !== sessionId),
    }));
  },
});