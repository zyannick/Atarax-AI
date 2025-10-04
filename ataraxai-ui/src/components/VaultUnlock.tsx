import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { Lock, Eye, EyeOff, RotateCcw } from 'lucide-react';
import { AtaraxLogo } from './AtaraxLogo';

interface VaultUnlockProps {
  onUnlock: (password: string) => Promise<boolean>;
  onReinitRequest: () => void;
  isReinitMode: boolean;
}

export function VaultUnlock({ onUnlock, onReinitRequest, isReinitMode }: VaultUnlockProps) {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [error, setError] = useState('');
  const [reinitInput, setReinitInput] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!password) {
      setError('Please enter your vault password');
      return;
    }

    setIsUnlocking(true);
    
    try {
      const success = await onUnlock(password);
      if (!success) {
        setError('Incorrect password. Please try again.');
        setPassword('');
      }
    } catch (error) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsUnlocking(false);
    }
  };

  const handleReinitCheck = () => {
    if (reinitInput.toLowerCase().trim() === 'reset ataraxai vault') {
      onReinitRequest();
      setReinitInput('');
    } else {
      setError('Please enter the exact phrase: "reset ataraxai vault"');
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <AtaraxLogo className="h-16 w-16" />
          </div>
          <div>
            <h1 className="text-3xl tracking-tight">Atarax-AI</h1>
            <p className="text-muted-foreground">Secure Local AI Assistant</p>
          </div>
        </div>

        <Card>
          <CardHeader className="text-center">
            <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mb-2">
              <Lock className="w-6 h-6 text-primary" />
            </div>
            <CardTitle>Unlock Vault</CardTitle>
            <CardDescription>
              Enter your vault password to access your AI assistant
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            {!isReinitMode ? (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="password">Vault Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Enter your vault password"
                      className="pr-10"
                      disabled={isUnlocking}
                      autoFocus
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3"
                      onClick={() => setShowPassword(!showPassword)}
                      disabled={isUnlocking}
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={isUnlocking || !password}
                >
                  {isUnlocking ? 'Unlocking...' : 'Unlock Vault'}
                </Button>
              </form>
            ) : (
              <div className="space-y-4">
                <Alert>
                  <RotateCcw className="w-4 h-4" />
                  <AlertDescription>
                    To reset your vault, type: <strong>reset ataraxai vault</strong>
                  </AlertDescription>
                </Alert>
                
                <div className="space-y-2">
                  <Label htmlFor="reinit">Confirmation Phrase</Label>
                  <Input
                    id="reinit"
                    value={reinitInput}
                    onChange={(e) => setReinitInput(e.target.value)}
                    placeholder="Type the confirmation phrase"
                    autoFocus
                  />
                </div>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <div className="flex gap-2">
                  <Button 
                    variant="outline"
                    onClick={() => {
                      onReinitRequest();
                      setReinitInput('');
                      setError('');
                    }}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleReinitCheck}
                    disabled={!reinitInput}
                    className="flex-1"
                  >
                    Reset Vault
                  </Button>
                </div>
              </div>
            )}

            {!isReinitMode && (
              <div className="mt-6 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    onReinitRequest();
                    setError('');
                  }}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  <RotateCcw className="w-3 h-3 mr-1" />
                  Reset Vault
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}