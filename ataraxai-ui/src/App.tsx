import { useState, useEffect } from "react";
import { AppProvider, useAppStore } from "./store/AppContext";
import { LeftSidebar } from "./components/LeftSidebar";
import { RightSidebar } from "./components/RightSidebar";
import { ChatView } from "./components/ChatView";
import { RAGSettingsView } from "./components/RAGSettingsView";
import { ModelManagerView } from "./components/ModelManagerView";
import { BenchmarkView } from "./components/BenchmarkView";
import { VaultSetup } from "./components/VaultSetup";
import { VaultUnlock } from "./components/VaultUnlock";
import { VaultReinit } from "./components/VaultReinit";
import { getAppStatus, initializeVault, unlockVault, API_STATUS, lockVault } from './lib/api';
import { AtaraxLogo } from "./components/AtaraxLogo";

type BackendState = "FIRST_LAUNCH" | "LOCKED" | "UNLOCKED" | "READY" | "ERROR";

type AppStatus = "loading" | "ready" | "error";
type VaultState = "uninitialized" | "locked" | "unlocked" | "reinit";

function AppContent() {

  const { currentView, fetchInitialData } = useAppStore();


  const [appStatus, setAppStatus] = useState<AppStatus>("loading");
  const [vaultState, setVaultState] = useState<VaultState>("locked");
  const [statusError, setStatusError] = useState<string | null>(null);
  const [connectionProgress, setConnectionProgress] = useState({
    elapsed: 0,
    attempt: 0,
    status: "Initializing backend",
  });

  useEffect(() => {

    const checkBackendStatus = async () => {
      const MAX_ATTEMPTS = 50;
      const INITIAL_INTERVAL = 2000;
      const MAX_INTERVAL = 5000;

      console.log("Starting backend connection...");

      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        const elapsed = Math.floor(((attempt - 1) * INITIAL_INTERVAL) / 1000);
        
        let statusMessage = "Initializing backend";
        if (elapsed >= 30) statusMessage = "Starting Python backend";
        if (elapsed >= 50) statusMessage = "Loading AI models";
        if (elapsed >= 70) statusMessage = "Establishing connection";
        if (elapsed >= 90) statusMessage = "This is taking longer than expected";

        setConnectionProgress({ elapsed, attempt, status: statusMessage });

        console.log(
          `Connection attempt ${attempt}/${MAX_ATTEMPTS} (${elapsed}s elapsed)`,
        );

        try {
          const response = await getAppStatus();
          if (response.status === API_STATUS.SUCCESS) {
            console.log("Received successful response from backend:", response);
            const backendStateString = response.message.split(
              ": ",
            )[1] as BackendState;
            console.log(
              `Backend connected! Initial state: ${backendStateString}`,
            );

            switch (backendStateString) {
              case "FIRST_LAUNCH":
              case "ERROR":
                setVaultState("uninitialized");
                break;
              case "LOCKED":
                setVaultState("locked");
                break;
              case "UNLOCKED":
              case "READY":
                setVaultState("unlocked");
                break;
              default:
                console.error(`Unknown backend state: ${backendStateString}`);
                setStatusError(
                  `Unexpected backend state: ${backendStateString}`,
                );
                setAppStatus("error");
                return;
            }
            setAppStatus("ready");
            return;
          }

          console.warn(
            `Backend not ready (attempt ${attempt}/${MAX_ATTEMPTS}). Response:`,
            response,
          );
        } catch (error) {
          console.warn(`Connection attempt ${attempt} failed:`, error);

          if (attempt > MAX_ATTEMPTS / 2) {
            console.error(
              "Connection attempts consistently failing. This may indicate a backend startup issue.",
            );
          }
        }

        if (attempt < MAX_ATTEMPTS) {
          const interval = Math.min(
            INITIAL_INTERVAL + Math.floor(attempt / 5) * 1000,
            MAX_INTERVAL,
          );
          await new Promise((resolve) => setTimeout(resolve, interval));
        }
      }

      const totalTime = Math.floor((MAX_ATTEMPTS * INITIAL_INTERVAL) / 1000);
      setStatusError(
        `Could not establish connection to backend after ${MAX_ATTEMPTS} attempts (${totalTime}s). ` +
          `The backend may have failed to start. Check the console for errors.`,
      );
      setAppStatus("error");
    };

    checkBackendStatus();
  }, []);


  useEffect(() => {
    if (vaultState === "unlocked" && appStatus === "ready") {
      console.log("Vault is unlocked, fetching initial data...");

      fetchInitialData().catch((err) => {
        console.error("Failed to fetch initial data:", err);
      });
    }
  }, [vaultState, appStatus, fetchInitialData]);


  const handleInitializeVault = async (password: string): Promise<boolean> => {
    try {
      const response = await initializeVault(password);
      if (response.status === API_STATUS.SUCCESS) {
        setVaultState("unlocked");
        return true;
      }
      throw new Error(response.message || "Initialization failed.");
    } catch (error) {
      throw error;
    }
  };

  const handleUnlockVault = async (password: string): Promise<boolean> => {
    try {
      const response = await unlockVault(password);
      if (response.status === API_STATUS.SUCCESS) {
        setVaultState("unlocked");
        return true;
      }
      throw new Error(response.message || "Unlock failed.");
    } catch (error) {
      throw error;
    }
  };

  const handleLockVault = () => {
    try {
      lockVault();
    } catch (error) {
      console.error("Error locking vault:", error);
    }
    setVaultState("locked");
  };

  const handleReinitRequest = () =>
    setVaultState(vaultState === "reinit" ? "locked" : "reinit");
  const handleCancelReinit = () => setVaultState("locked");
  const handleReinitializeVault = async (
    currentPassword: string,
    newPassword: string,
  ): Promise<boolean> => {
    const unlocked = await handleUnlockVault(currentPassword);
    if (unlocked) {
      const reinitialized = await handleInitializeVault(newPassword);
      if (reinitialized) {
        setVaultState("unlocked");
        return true;
      }
    }
    return false;
  };


  if (appStatus === "loading") {
    const progress = Math.min((connectionProgress.elapsed / 60) * 100, 100);
    
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-off-white">
        <AtaraxLogo className="h-20 w-20 animate-pulse" />
        
        <div className="mt-6 text-center">
          <p className="text-lg text-cognac font-medium">
            {connectionProgress.status}
            <span className="inline-block w-8 text-left">
              {".".repeat((connectionProgress.elapsed % 4))}
            </span>
          </p>
          
          <div className="mt-4 flex items-center gap-2 justify-center">
            <div className="w-48 h-1.5 bg-cognac/20 rounded-full overflow-hidden">
              <div 
                className="h-full bg-cognac transition-all duration-1000 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-sm text-cognac/70 w-12 text-right font-mono">
              {connectionProgress.elapsed}s
            </span>
          </div>
          
          {connectionProgress.elapsed > 30 && (
            <p className="mt-4 text-xs text-cognac/70 max-w-md px-4">
              The backend is taking longer than usual to start. This can happen on first launch
              or if the system is under heavy load. Please wait...
            </p>
          )}
        </div>
      </div>
    );
  }

  if (appStatus === "error") {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-off-white text-center p-4">
        <AtaraxLogo className="h-20 w-20" />
        <h2 className="mt-4 text-xl text-red-600">Connection Error</h2>
        <p className="mt-2 text-gray-600">
          Could not connect to the AtaraxAI backend.
        </p>
        <p className="mt-1 text-sm text-gray-500">{statusError}</p>
        <p className="mt-4 text-xs text-gray-500">
          Please ensure the backend is running and try restarting the
          application.
        </p>
      </div>
    );
  }

  if (vaultState === "uninitialized") {
    return <VaultSetup onInitialize={handleInitializeVault} />;
  }

  if (vaultState === "locked") {
    return (
      <VaultUnlock
        onUnlock={handleUnlockVault}
        onReinitRequest={handleReinitRequest}
        isReinitMode={false}
      />
    );
  }

  if (vaultState === "reinit") {
    return (
      <VaultReinit
        onReinitialize={handleReinitializeVault}
        onCancel={handleCancelReinit}
      />
    );
  }

  const renderMainContent = () => {
    switch (currentView) {
      case "chat":
        return <ChatView />;
      case "rag-settings":
        return <RAGSettingsView />;
      case "model-manager":
        return <ModelManagerView />;
      case "benchmark":
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
        {<RightSidebar />}
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