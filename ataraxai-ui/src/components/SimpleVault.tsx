import React, { useState } from 'react';

interface SimpleVaultProps {
  onUnlock: () => void;
}

export function SimpleVault({ onUnlock }: SimpleVaultProps) {
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password === 'demo') {
      onUnlock();
    } else {
      alert('Password should be "demo"');
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <h1 className="text-2xl mb-6 text-center">Atarax-AI Vault</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="password" className="block text-sm mb-2">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password (hint: demo)"
              className="w-full p-2 border rounded"
            />
          </div>
          <button
            type="submit"
            className="w-full p-2 bg-primary text-primary-foreground rounded"
          >
            Unlock
          </button>
        </form>
      </div>
    </div>
  );
}