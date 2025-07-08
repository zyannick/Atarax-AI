// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ChatRequest {
    user_query: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    assistant_response: String,
}

#[tauri::command]
async fn send_chat_message(session_id: String, user_query: String) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:8000/v1/sessions/{}/messages", session_id);

    let response = client.post(&url)
        .json(&ChatRequest { user_query })
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if response.status().is_success() {
        let chat_response = response.json::<ChatResponse>().await.map_err(|e| e.to_string())?;
        Ok(chat_response.assistant_response)
    } else {
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        Err(format!("API Error: {}", error_text))
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![send_chat_message])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}