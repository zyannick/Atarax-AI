import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { BrainCircuit, Folder, MessageSquare, Plus, Send, X, Settings, Sun, Moon } from 'lucide-react';

const ChatView = ({ toggleTheme, currentTheme }: { toggleTheme: () => void; currentTheme: string }) => {
    const { messages, sendMessage, selectedSessionId, isLoading } = useAppStore();
    const [input, setInput] = useState('');

    const handleSend = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !selectedSessionId) return;
        
        await sendMessage(selectedSessionId, input);
        setInput('');
    };

    if (!selectedSessionId) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-4">
                <MessageSquare size={48} className="text-gray-300 dark:text-gray-600 mb-4" />
                <h3 className="font-semibold text-lg">No Session Selected</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Choose a session to start chatting.</p>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700/50">
                <h2 className="text-xl font-semibold">Chat</h2>
                <div className="flex items-center gap-2">
                    <button onClick={toggleTheme} className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
                        {currentTheme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
                    </button>
                    <button className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
                        <Settings size={20} />
                    </button>
                </div>
            </div>

            {/* Message List */}
            <div className="flex-1 p-6 overflow-y-auto">
                <div className="space-y-6">
                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex items-end gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role !== 'user' && (
                                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white flex-shrink-0">
                                    <BrainCircuit size={20} />
                                </div>
                            )}
                            <div className={`max-w-lg px-4 py-3 rounded-2xl ${
                                msg.role === 'user' 
                                    ? 'bg-blue-500 text-white rounded-br-none' 
                                    : msg.role === 'error'
                                    ? 'bg-red-100 dark:bg-red-500/20 text-red-700 dark:text-red-300 rounded-bl-none'
                                    : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-none'
                            }`}>
                                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        </div>
                    ))}
                    {isLoading && <div className="flex justify-start"><p className="text-sm text-gray-400">Assistant is typing...</p></div>}
                </div>
            </div>

            {/* Message Input */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700/50">
                <form onSubmit={handleSend} className="relative">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend(e);
                            }
                        }}
                        placeholder="Type your message..."
                        rows={1}
                        className="w-full p-3 pr-12 rounded-xl border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 resize-none focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition"
                    />
                    <button type="submit" className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50 transition-colors">
                        <Send size={18} />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatView;