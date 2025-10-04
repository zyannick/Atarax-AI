import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { Shield, Eye, EyeOff } from 'lucide-react';
import { AtaraxLogo } from './AtaraxLogo';

interface VaultSetupProps {
  onInitialize: (password: string) => Promise<boolean>;
}

export function VaultSetup({ onInitialize }: VaultSetupProps) {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setIsInitializing(true);
    
    try {
      const success = await onInitialize(password);
      if (!success) {
        setError('Failed to initialize vault. Please try again.');
      }
    } catch (error) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsInitializing(false);
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
              <Shield className="w-6 h-6 text-primary" />
            </div>
            <CardTitle>Initialize Vault</CardTitle>
            <CardDescription>
              Create a secure password to protect your AI assistant and data
            </CardDescription>
          </CardHeader>
          
          <CardContent>
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
                    disabled={isInitializing}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3"
                    onClick={() => setShowPassword(!showPassword)}
                    disabled={isInitializing}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm Password</Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm your vault password"
                    className="pr-10"
                    disabled={isInitializing}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    disabled={isInitializing}
                  >
                    {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
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
                disabled={isInitializing || !password || !confirmPassword}
              >
                {isInitializing ? 'Initializing...' : 'Initialize Vault'}
              </Button>
            </form>

            <div className="mt-6 text-sm text-muted-foreground space-y-2">
              <p><strong>Security Features:</strong></p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>End-to-end encryption for all data</li>
                <li>Local storage - no cloud dependencies</li>
                <li>Auto-lock after inactivity</li>
                <li>Secure vault reinitialization</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}