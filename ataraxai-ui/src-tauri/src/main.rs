// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tauri::{State};
use tokio::sync::RwLock;
use uuid::Uuid;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub project_id: String,
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub title: String,
    pub project_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: String, // "user" | "assistant" | "error"
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
    user_query: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    assistant_response: String,
}

#[derive(Debug)]
pub struct AppError {
    pub message: String,
    pub kind: ErrorKind,
}

#[derive(Debug)]
pub enum ErrorKind {
    Network,
    Database,
    Validation,
    NotFound,
    Internal,
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind_str(), self.message)
    }
}

impl AppError {
    fn kind_str(&self) -> &'static str {
        match self.kind {
            ErrorKind::Network => "Network Error",
            ErrorKind::Database => "Database Error", 
            ErrorKind::Validation => "Validation Error",
            ErrorKind::NotFound => "Not Found",
            ErrorKind::Internal => "Internal Error",
        }
    }
}


#[derive(Debug)]
pub struct AppState {
    pub projects: Arc<RwLock<HashMap<String, Project>>>,
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub messages: Arc<RwLock<HashMap<String, Vec<Message>>>>,
    pub http_client: reqwest::Client,
    pub api_base_url: String,
}

impl AppState {
    pub fn new() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            projects: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            messages: Arc::new(RwLock::new(HashMap::new())),
            http_client,
            api_base_url: "http://127.0.0.1:8000/v1".to_string(),
        }
    }
}


fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

fn validate_string(input: &str, field_name: &str, min_length: usize) -> Result<(), AppError> {
    if input.trim().is_empty() {
        return Err(AppError {
            message: format!("{} cannot be empty", field_name),
            kind: ErrorKind::Validation,
        });
    }
    if input.trim().len() < min_length {
        return Err(AppError {
            message: format!("{} must be at least {} characters long", field_name, min_length),
            kind: ErrorKind::Validation,
        });
    }
    Ok(())
}


#[tauri::command]
async fn list_projects(state: State<'_, AppState>) -> Result<Vec<Project>, String> {
    let projects = state.projects.read().await;
    Ok(projects.values().cloned().collect())
}

#[tauri::command]
async fn create_project(
    name: String,
    description: Option<String>,
    state: State<'_, AppState>,
) -> Result<Project, String> {
    validate_string(&name, "Project name", 1)
        .map_err(|e| e.to_string())?;

    let project = Project {
        project_id: generate_id(),
        name: name.trim().to_string(),
        description: description.map(|d| d.trim().to_string()),
    };

    let mut projects = state.projects.write().await;
    projects.insert(project.project_id.clone(), project.clone());
    
    Ok(project)
}

#[tauri::command]
async fn delete_project(
    project_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut projects = state.projects.write().await;
    let mut sessions = state.sessions.write().await;
    let mut messages = state.messages.write().await;

    projects.remove(&project_id)
        .ok_or_else(|| "Project not found".to_string())?;

    sessions.retain(|_, session| session.project_id != project_id);

    let session_ids: Vec<String> = sessions
        .values()
        .filter(|s| s.project_id == project_id)
        .map(|s| s.session_id.clone())
        .collect();

    for session_id in session_ids {
        messages.remove(&session_id);
    }

    Ok(())
}


#[tauri::command]
async fn list_sessions(
    project_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<Session>, String> {
    let sessions = state.sessions.read().await;
    Ok(sessions
        .values()
        .filter(|s| s.project_id == project_id)
        .cloned()
        .collect())
}

#[tauri::command]
async fn create_session(
    project_id: String,
    title: String,
    state: State<'_, AppState>,
) -> Result<Session, String> {
    validate_string(&title, "Session title", 1)
        .map_err(|e| e.to_string())?;

    let projects = state.projects.read().await;
    if !projects.contains_key(&project_id) {
        return Err("Project not found".to_string());
    }
    drop(projects);

    let session = Session {
        session_id: generate_id(),
        title: title.trim().to_string(),
        project_id,
    };

    let mut sessions = state.sessions.write().await;
    sessions.insert(session.session_id.clone(), session.clone());
    
    Ok(session)
}

#[tauri::command]
async fn delete_session(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut sessions = state.sessions.write().await;
    let mut messages = state.messages.write().await;

    sessions.remove(&session_id)
        .ok_or_else(|| "Session not found".to_string())?;

    messages.remove(&session_id);
    
    Ok(())
}


#[tauri::command]
async fn list_messages(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<Message>, String> {
    let messages = state.messages.read().await;
    Ok(messages
        .get(&session_id)
        .cloned()
        .unwrap_or_default())
}

#[tauri::command]
async fn send_message(
    session_id: String,
    user_query: String,
    state: State<'_, AppState>,
) -> Result<Message, String> {
    validate_string(&user_query, "Message", 1)
        .map_err(|e| e.to_string())?;

    let sessions = state.sessions.read().await;
    if !sessions.contains_key(&session_id) {
        return Err("Session not found".to_string());
    }
    drop(sessions);

    let user_message = Message {
        id: generate_id(),
        role: "user".to_string(),
        content: user_query.clone(),
    };

    {
        let mut messages = state.messages.write().await;
        messages
            .entry(session_id.clone())
            .or_insert_with(Vec::new)
            .push(user_message);
    }

    let url = format!("{}/sessions/{}/messages", state.api_base_url, session_id);
    let response = state
        .http_client
        .post(&url)
        .json(&ChatRequest { user_query })
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;

    let assistant_message = if response.status().is_success() {
        let chat_response = response
            .json::<ChatResponse>()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;
        
        Message {
            id: generate_id(),
            role: "assistant".to_string(),
            content: chat_response.assistant_response,
        }
    } else {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        
        Message {
            id: generate_id(),
            role: "error".to_string(),
            content: format!("API Error ({}): {}", status, error_text),
        }
    };

    {
        let mut messages = state.messages.write().await;
        messages
            .entry(session_id)
            .or_insert_with(Vec::new)
            .push(assistant_message.clone());
    }

    Ok(assistant_message)
}

#[tauri::command]
async fn clear_messages(
    session_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut messages = state.messages.write().await;
    messages.insert(session_id, Vec::new());
    Ok(())
}

#[tauri::command]
async fn get_api_config(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.api_base_url.clone())
}

#[tauri::command]
async fn update_api_config(
    new_url: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    validate_string(&new_url, "API URL", 1)
        .map_err(|e| e.to_string())?;

    if !new_url.starts_with("http://") && !new_url.starts_with("https://") {
        return Err("API URL must start with http:// or https://".to_string());
    }

    let test_url = format!("{}/health", new_url);
    let response = state
        .http_client
        .get(&test_url)
        .send()
        .await
        .map_err(|e| format!("Cannot connect to API: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("API health check failed: {}", response.status()));
    }

    
    Ok(())
}


fn main() {
    let app_state = AppState::new();

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            list_projects,
            create_project,
            delete_project,
            list_sessions,
            create_session,
            delete_session,
            list_messages,
            send_message,
            clear_messages,
            get_api_config,
            update_api_config,
        ])
        .setup(|app| {
            println!("Application started successfully");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

