import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';

export const ChatWindow = () => {
  const { messages, selectedSessionId, sendMessage, isLoading, error } = useAppStore();
  const [inputValue, setInputValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input when session changes
  useEffect(() => {
    if (selectedSessionId && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selectedSessionId]);

  if (!selectedSessionId) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <div className="text-2xl mb-2">💬</div>
          <p>Select a session to start chatting</p>
          <p className="text-sm mt-1">Choose from the sessions panel or create a new one</p>
        </div>
      </div>
    );
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      await sendMessage(selectedSessionId, inputValue);
      setInputValue('');
    } catch (error) {
      // Error is already handled in the store
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col p-4">
      {/* Error Banner */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
          <p className="text-red-800 dark:text-red-200 text-sm">{error}</p>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto space-y-4 p-2 rounded-md bg-white dark:bg-gray-800/50">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-4">🤖</div>
              <p>No messages yet</p>
              <p className="text-sm mt-1">Start a conversation below!</p>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-lg p-3 rounded-2xl ${
                msg.role === 'user' 
                  ? 'bg-blue-500 text-white rounded-br-none' 
                  : msg.role === 'error'
                  ? 'bg-red-500 text-white rounded-bl-none'
                  : 'bg-gray-200 dark:bg-gray-700 rounded-bl-none'
              }`}>
                <div className="whitespace-pre-wrap">{msg.content}</div>
              </div>
            </div>
          ))
        )}
        
        {/* Loading indicator */}
        {isSubmitting && (
          <div className="flex justify-start">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-2xl rounded-bl-none p-3">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSendMessage} className="mt-4 flex items-center">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
          disabled={isSubmitting}
          className="flex-1 p-3 rounded-lg border-2 border-gray-300 dark:border-gray-600 focus:outline-none focus:border-blue-500 dark:bg-gray-800 disabled:opacity-50"
          aria-label="Chat message input"
        />
        <button 
          type="submit" 
          disabled={!inputValue.trim() || isSubmitting}
          className="ml-3 p-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          aria-label="Send message"
        >
          {isSubmitting ? '⏳' : 'Send'}
        </button>
      </form>
    </div>
  );
};
