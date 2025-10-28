import { invoke } from "@tauri-apps/api/core";

interface ApiInfo {
  port: number;
  token: string;
  status: string;
}

interface ApiResponse {
  status: string;
  message: string;
  data?: any;
}

export const API_STATUS = {
    SUCCESS: "success",
    FAILURE: "failure",
};


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

const apiClient = new ApiClient();

/**
 * A generic helper function for making authenticated fetch requests to the Python backend.
 * @param endpoint The API endpoint to call (e.g., "/api/v1/vault/init")
 * @param options The standard Fetch API options object (method, body, etc.)
 * @returns The JSON response from the API.
 */
async function apiFetch(
  endpoint: string,
  options: RequestInit = {},
): Promise<ApiResponse> {
  const clientInfo = await apiClient.getClient();
  if (!clientInfo) {
    throw new Error("Backend API connection is not available.");
  }

  const { port, token } = clientInfo;
  console.log(`Making API call to ${endpoint} on port ${port}`);
  const url = `http://127.0.0.1:${port}${endpoint}`;

  const headers = {
    ...options.headers,
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  const response = await fetch(url, { ...options, headers });

  console.log(`Received response from ${endpoint}:`, response);

  if (!response.ok) {
    let errorDetail = `HTTP error! Status: ${response.status}`;
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorDetail;
    } catch (e) {
    }
    throw new Error(errorDetail);
  }

  const data: ApiResponse = await response.json();
  console.log(`Received successful parsed response from ${endpoint}:`, data);
  return data;
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
  return apiFetch("/api/v1/vault/initialize", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

export async function lockVault(): Promise<ApiResponse> {
  return apiFetch("/api/v1/vault/lock", {
    method: "POST",
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

export async function createProject(
  name: string, 
  description: string
): Promise<ApiResponse> {
  return apiFetch("/api/v1/chat/projects", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function listProjects(): Promise<ApiResponse> {
  return apiFetch("/api/v1/chat/projects");
}

export async function createChatSession(
  projectId: string,
): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/projects/${projectId}/sessions`, {
    method: "POST",
  });
}

export async function addRagDirectories(paths: string[]): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/add_directories", {
    method: "POST",
    body: JSON.stringify({ directory_paths: paths }),
  });
}

export async function removeRagDirectories(
  paths: string[],
): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/remove_directories", {
    method: "POST",
    body: JSON.stringify({ directory_paths: paths }),
  });
}

export async function listRagDirectories(): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/list_directories");
}

export async function getRagStatus(): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/health");
}

export async function rebuildRagIndex(): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/rebuild_index", { method: "POST" });
}

export async function checkRagManifest(): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/check_manifest");
}

export async function ragHealthCheck(): Promise<ApiResponse> {
  return apiFetch("/api/v1/rag/health_check");
}

export async function listLocalModels(): Promise<ApiResponse> {
  return apiFetch("/api/v1/models_manager/models");
}

export async function downloadModel(
  repoId: string,
  fileName: string,
): Promise<ApiResponse> {
  return apiFetch("/api/v1/models_manager/download", {
    method: "POST",
    body: JSON.stringify({ repo_id: repoId, file_name: fileName }),
  });
}

export async function getDownloadStatus(taskId: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/models_manager/download/status/${taskId}`);
}

export interface QuantizedModelInfo {
  modelID: string;
  fileName: string;
  lastModified?: string;
  quantization?: string;
  fileSize?: number;
}

export interface BenchmarkParams {
  n_gpu_layers?: number;
  n_gen?: number;
  repetitions?: number;
  warmup?: boolean;
  temperature?: number;
  top_k?: number;
  top_p?: number;
}

export interface LlamaModelParams {
  model_path: string;
  n_ctx?: number;
  n_gpu_layers?: number;
}

export interface BenchmarkJobPayload {
  model_info: QuantizedModelInfo;
  benchmark_params: BenchmarkParams;
  llama_model_params: LlamaModelParams;
}

export async function enqueueBenchmarkJob(
  payload: BenchmarkJobPayload,
): Promise<ApiResponse> {
  return apiFetch("/api/v1/benchmarker/jobs/enqueue", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getBenchmarkJobStatus(
  jobId: string,
): Promise<ApiResponse> {
  return apiFetch(`/api/v1/benchmarker/job/${jobId}`);
}

export async function getBenchmarkQueueStatus(): Promise<ApiResponse> {
  return apiFetch("/api/v1/benchmarker/status");
}


export async function getUserPreferences(): Promise<ApiResponse> {
  return apiFetch("/api/v1/user-preferences");
}

export async function updateUserPreferences(
  preferences: any,
): Promise<ApiResponse> {
  return apiFetch("/api/v1/user-preferences", {
    method: "PUT",
    body: JSON.stringify(preferences),
  });
}
