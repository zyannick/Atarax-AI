import { invoke } from "@tauri-apps/api/core";

interface ApiInfo {
  port: number;
  token: string;
  status: string;
}

interface ApiResponse {
    status: "SUCCESS" | "FAILURE";
    message: string;
    data?: any;
}

/**
 * A singleton class to manage the connection details to the Python backend.
 * It fetches the details once from Rust and then provides them for all subsequent API calls.
 */
class ApiClient {
  private apiInfo: Promise<ApiInfo | null>;

  constructor() {
    this.apiInfo = this.fetchApiInfo();
  }

  private async fetchApiInfo(): Promise<ApiInfo | null> {
    try {
      // This is the one-time call to the Rust backend to get connection details.
      const info = await invoke<ApiInfo>("get_api_info");
      console.log("Successfully fetched API info from Rust:", info);
      return info;
    } catch (error) {
      console.error("Failed to fetch API info from Rust:", error);
      return null;
    }
  }

  public async getClient() {
    return this.apiInfo;
  }
}

// Create a single, shared instance of the ApiClient
const apiClient = new ApiClient();

/**
 * A generic helper function for making authenticated fetch requests to the Python backend.
 * @param endpoint The API endpoint to call (e.g., "/api/v1/vault/init")
 * @param options The standard Fetch API options object (method, body, etc.)
 * @returns The JSON response from the API.
 */
async function apiFetch(endpoint: string, options: RequestInit = {}): Promise<ApiResponse> {
  const clientInfo = await apiClient.getClient();
  if (!clientInfo) {
    throw new Error("Backend API connection is not available.");
  }

  const { port, token } = clientInfo;
  const url = `http://127.0.0.1:${port}${endpoint}`;

  const headers = {
    ...options.headers,
    "Content-Type": "application/json",
    "Authorization": `Bearer ${token}`,
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    let errorDetail = `HTTP error! Status: ${response.status}`;
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorDetail;
    } catch (e) {
      // Ignore if the response is not JSON
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

/**
 * Fetches the current status of the backend application from the orchestrator.
 * This is used on startup to determine which UI to show (e.g., setup, unlock, or main app).
 * @returns The API response, with the application state (e.g., "FIRST_LAUNCH", "LOCKED") in the `data` field.
 */
export async function getAppStatus(): Promise<ApiResponse> {
    return apiFetch("/v1/status");
}


/**
 * Calls the backend to initialize the secure vault with a password.
 * @param password The user's chosen password.
 * @returns The API response.
 */
export async function initializeVault(password: string): Promise<ApiResponse> {
    return apiFetch("/api/v1/vault/init", {
        method: "POST",
        body: JSON.stringify({ password }),
    });
}

/**
 * Calls the backend to unlock the secure vault with a password.
 * @param password The user's password.
 * @returns The API response.
 */
export async function unlockVault(password: string): Promise<ApiResponse> {
    return apiFetch("/api/v1/vault/unlock", {
        method: "POST",
        body: JSON.stringify({ password }),
    });
}


export async function sendChatMessage(message: string): Promise<ApiResponse> {
  return apiFetch("/api/v1/chat/message", {
      method: "POST",
      body: JSON.stringify({ text: message }),
  });
}

