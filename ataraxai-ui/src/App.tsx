import React, { useState, useEffect } from 'react';
import { AppProvider, useAppStore } from './store/AppContext';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { ChatView } from './components/ChatView';
import { RAGSettingsView } from './components/RAGSettingsView';
import { ModelManagerView } from './components/ModelManagerView';
import { BenchmarkView } from './components/BenchmarkView';
import { VaultSetup } from './components/VaultSetup';
import { VaultUnlock } from './components/VaultUnlock';
import { VaultReinit } from './components/VaultReinit';

// Vault utilities
const STORAGE_KEY = 'atarax_vault_hash';

const hashPassword = (password: string): string => {
  let hash = 0;
  for (let i = 0; i < password.length; i++) {
    const char = password.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString();
};

const getVaultHash = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

const setVaultHash = (hash: string): boolean => {
  try {
    localStorage.setItem(STORAGE_KEY, hash);
    return true;
  } catch {
    return false;
  }
};

type VaultState = 'uninitialized' | 'locked' | 'unlocked' | 'reinit';

function AppContent() {
  const { currentView } = useAppStore();
  
  // Initialize vault state based on localStorage
  const [vaultState, setVaultState] = useState<VaultState>(() => {
    const storedHash = getVaultHash();
    if (!storedHash) return 'uninitialized';
    return 'locked';
  });

  // Vault handlers
  const handleInitializeVault = async (password: string): Promise<boolean> => {
    if (password.length < 6) return false;
    
    const hash = hashPassword(password);
    const success = setVaultHash(hash);
    
    if (success) {
      setVaultState('unlocked');
    }
    
    return success;
  };

  const handleUnlockVault = async (password: string): Promise<boolean> => {
    const storedHash = getVaultHash();
    if (!storedHash) return false;
    
    const inputHash = hashPassword(password);
    const success = inputHash === storedHash;
    
    if (success) {
      setVaultState('unlocked');
    }
    
    return success;
  };

  const handleLockVault = () => {
    setVaultState('locked');
  };

  const handleReinitRequest = () => {
    if (vaultState === 'reinit') {
      setVaultState('locked');
    } else {
      setVaultState('reinit');
    }
  };

  const handleReinitializeVault = async (currentPassword: string, newPassword: string): Promise<boolean> => {
    const storedHash = getVaultHash();
    if (!storedHash) return false;
    
    const currentHash = hashPassword(currentPassword);
    if (currentHash !== storedHash) return false;
    
    if (newPassword.length < 6) return false;
    
    const newHash = hashPassword(newPassword);
    const success = setVaultHash(newHash);
    
    if (success) {
      setVaultState('unlocked');
    }
    
    return success;
  };

  const handleCancelReinit = () => {
    setVaultState('locked');
  };

  // Vault UI states
  if (vaultState === 'uninitialized') {
    return <VaultSetup onInitialize={handleInitializeVault} />;
  }

  if (vaultState === 'locked') {
    return (
      <VaultUnlock
        onUnlock={handleUnlockVault}
        onReinitRequest={handleReinitRequest}
        isReinitMode={false}
      />
    );
  }

  if (vaultState === 'reinit') {
    return (
      <VaultReinit
        onReinitialize={handleReinitializeVault}
        onCancel={handleCancelReinit}
      />
    );
  }

  // Main app UI (unlocked state)
  const renderMainContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatView />;
      case 'rag-settings':
        return <RAGSettingsView />;
      case 'model-manager':
        return <ModelManagerView />;
      case 'benchmark':
        return <BenchmarkView />;
      default:
        return <ChatView />;
    }
  };

  return (
    <div className="h-screen w-screen bg-background text-foreground overflow-hidden">
      <div className="flex h-full">
        <LeftSidebar onLockVault={handleLockVault} />
        {renderMainContent()}
        <RightSidebar />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}