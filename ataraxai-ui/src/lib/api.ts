import { invoke } from "@tauri-apps/api/core";

interface ApiInfo {
  port: number;
  token: string;
  status: string;
}

interface ApiResponse {
  status: string;
  message: string;
  [key: string]: any;
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
 * @param endpoint The API endpoint to call
 * @param options The standard Fetch API options object
 * @returns The JSON response from the API, assumed to be ApiResponse.
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
  console.log(`Making API call to ${endpoint} on port ${port} with method ${options.method || 'GET'}`);
  const url = `http://127.0.0.1:${port}${endpoint}`;

  const headers = {
    ...options.headers,
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  const response = await fetch(url, { ...options, headers });

  console.log(`Received response from ${endpoint}:`, response.status, response.statusText);

  const contentType = response.headers.get("content-type");
  let data: ApiResponse | null = null;
  if (contentType && contentType.includes("application/json")) {
      try {
        data = await response.json();
        console.log(`Received successful parsed response from ${endpoint}:`, data);
      } catch (e) {
         console.error(`Failed to parse JSON response from ${endpoint}:`, e);
         if (response.ok) {
             data = { status: API_STATUS.SUCCESS, message: response.statusText || "Operation successful" };
         } else {
             throw new Error(`HTTP error! Status: ${response.status} ${response.statusText}`);
         }
      }
  } else if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status} ${response.statusText}`);
  } else {
      data = { status: API_STATUS.SUCCESS, message: response.statusText || "Operation successful" };
  }


  if (!response.ok && data) {
    const errorDetail = data.detail || `HTTP error! Status: ${response.status}`;
    throw new Error(errorDetail);
  } else if (!response.ok) {
       throw new Error(`HTTP error! Status: ${response.status} ${response.statusText}`);
  }

  if (data === null) {
      throw new Error("API response processing failed unexpectedly.");
  }

  return data;
}


export async function getAppStatus(): Promise<ApiResponse> {
  return apiFetch("/v1/status");
}

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

export async function createProject(name: string, description: string): Promise<ApiResponse> {
  return apiFetch("/api/v1/chat/projects", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function listProjects(): Promise<ApiResponse> {
  return apiFetch("/api/v1/chat/projects");
}

export async function updateProjectApi(id: string, name: string, description: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/projects/${id}`, {
    method: "PUT",
    body: JSON.stringify({ name, description }),
  });
}

export async function deleteProjectApi(id: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/projects/${id}`, {
    method: "DELETE",
  });
}


export async function createChatSession(
  projectId: string,
  title?: string,
): Promise<ApiResponse> {
  const body = title ? { project_id: projectId, title } : { project_id: projectId };
  return apiFetch(`/api/v1/chat/sessions`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listSessions(projectId: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/projects/${projectId}/sessions`);
}

export async function renameSessionApi(id: string, title: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/sessions/${id}`, {
    method: "PUT",
    body: JSON.stringify({ title }),
  });
}

export async function deleteSessionApi(id: string): Promise<ApiResponse> {
  return apiFetch(`/api/v1/chat/sessions/${id}`, {
    method: "DELETE",
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

