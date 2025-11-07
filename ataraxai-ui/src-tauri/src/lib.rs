// Prevents additional console window on Windows in release
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{async_runtime, AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;


#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiInfo {
    port: u16,
    token: String,
    status: String,
}


#[derive(Debug, Default)]
struct ApiState(Mutex<Option<ApiInfo>>);

pub struct ApiProcess(Mutex<Option<CommandChild>>);

impl ApiState {
    fn set_info(&self, info: ApiInfo) {
        let mut guard = self.0.lock().unwrap();
        *guard = Some(info);
    }

    fn get_info(&self) -> Option<ApiInfo> {
        let guard = self.0.lock().unwrap();
        guard.clone()
    }
}


#[tauri::command]
async fn get_api_info(state: State<'_, ApiState>) -> Result<ApiInfo, String> {
    let timeout_duration = Duration::from_secs(120);
    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(1000); 

    while start.elapsed() < timeout_duration {
        if let Some(info) = state.get_info() {
            println!("API connection details acquired after {:?}", start.elapsed());
            println!("Port: {}, Token: {}", info.port, info.token);
            return Ok(info);
        }
        
        let elapsed_secs = start.elapsed().as_secs();
        if elapsed_secs > 0 && elapsed_secs % 5 == 0 {
            println!("Still waiting for Python backend... ({}s elapsed)", elapsed_secs);
        }
        
        tokio::time::sleep(poll_interval).await;
    }

    let elapsed = start.elapsed();
    eprintln!(
        "API failed to provide connection details after {:?}. Python backend may have crashed or failed to start.",
        elapsed
    );
    
    Err(format!(
        "Backend startup timeout ({:.1}s). Check console logs for Python errors.",
        elapsed.as_secs_f32()
    ))
}

#[tauri::command]
fn stop_python_sidecar(state: State<'_, ApiProcess>) -> Result<(), String> {
    if let Some(child) = state.0.lock().unwrap().take() {
        child.kill().map_err(|e| format!("Failed to kill sidecar: {}", e))
    } else {
        Err("No sidecar process was running.".into())
    }
}


async fn start_python_sidecar(
    app_handle: AppHandle,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Resolving path for Python sidecar executable 'api'...");
    
    let api_state: State<ApiState> = app_handle.state();
    let api_process_state: State<ApiProcess> = app_handle.state();

    let executable_name = if cfg!(target_os = "windows") {
        "api.exe"
    } else {
        "api"
    };

    let resource_path = format!("py_src/{}", executable_name);
    
    let executable_path = app_handle
        .path()
        .resolve(&resource_path, tauri::path::BaseDirectory::Resource)?;
    
    println!("Starting Python sidecar from: {:?}", executable_path);

    let (mut rx, child) = app_handle.shell().command(&executable_path).spawn()?;
    
    *api_process_state.0.lock().unwrap() = Some(child);

    println!("Waiting for Python backend to emit connection details...");
    
    let mut handshake_complete = false;

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                if let Ok(line_str) = String::from_utf8(line) {
                    if !handshake_complete {
                        if let Ok(api_info) = serde_json::from_str::<ApiInfo>(&line_str) {
                            if api_info.status == "ready" {
                                println!("Backend is ready. Port: {}, Token acquired.", api_info.port);
                                api_state.set_info(api_info);
                                handshake_complete = true;
                            }
                        }
                    } else {
                        println!("Python sidecar (stdout): {}", line_str.trim());
                    }
                }
            }
            CommandEvent::Stderr(line) => {
                if let Ok(line_str) = String::from_utf8(line) {
                    eprintln!("Python sidecar (stderr): {}", line_str.trim());
                }
            }
            CommandEvent::Error(line) => {
                eprintln!("Python sidecar error: {:?}", line);
            }
            CommandEvent::Terminated(payload) => {
                eprintln!("Python sidecar terminated with status: {:?}", payload);
                if !handshake_complete {
                    return Err("Sidecar process terminated before it became ready.".into());
                }
                break; 
            }
            _ => {}
        }
    }
    
    Ok(())
}


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(ApiState::default())
        .manage(ApiProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            get_api_info,
            stop_python_sidecar
        ])
        .setup(|app| {
            let app_handle = app.handle().clone();
            async_runtime::spawn(async move {
                if let Err(e) = start_python_sidecar(app_handle.clone()).await {
                    let err_msg = format!("Failed to start Python sidecar: {}", e);
                    eprintln!("{}", err_msg);
                    let _ = app_handle.emit("sidecar-error", err_msg);
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                println!("Window closed, terminating sidecar process...");
                let state: State<ApiProcess> = window.state();
                let child_to_kill = state.0.lock().unwrap().take();
                if let Some(child) = child_to_kill {
                    if let Err(e) = child.kill() {
                        eprintln!("Failed to kill sidecar on exit: {}", e);
                    } else {
                        println!("Sidecar process terminated successfully.");
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}