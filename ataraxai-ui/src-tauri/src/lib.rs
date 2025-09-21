#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::{async_runtime, AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiInfo {
    status: String,
    token: String,
    port: Option<u16>,
}

#[derive(Debug, Default)]
struct ApiState(Mutex<Option<ApiInfo>>);

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

pub struct ApiProcess(Mutex<Option<CommandChild>>);

async fn start_python_sidecar(
    app_handle: AppHandle,
    api_state: State<'_, ApiState>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let executable_path = app_handle
        .path()
        .resolve("py_src/api", tauri::path::BaseDirectory::Resource)?;

    let (mut rx, child) = app_handle.shell().command(&executable_path).spawn()?;

    {
        let api_process_state: State<ApiProcess> = app_handle.state();
        let mut guard = api_process_state.0.lock().unwrap();
        *guard = Some(child);
    }

    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                if let Ok(text) = String::from_utf8(line) {
                    println!("Received from Python: {}", text);

                    if let Ok(api_info) = serde_json::from_str::<ApiInfo>(&text) {
                        if api_info.status == "ready" {
                            api_state.set_info(api_info);
                            return Ok(());
                        }
                    }
                } else {
                    eprintln!("Invalid UTF-8 from Python stdout");
                }
            }
            CommandEvent::Stderr(line) => {
                if let Ok(text) = String::from_utf8(line) {
                    eprintln!("Python sidecar (stderr): {}", text);
                }
            }
            CommandEvent::Error(line) => {
                eprintln!("Python sidecar error: {:?}", line);
            }
            _ => {}
        }
    }

    Err("Sidecar process closed before sending ready signal.".into())
}

#[tauri::command]
fn get_api_info(state: State<ApiState>) -> Option<ApiInfo> {
    state.get_info()
}

#[tauri::command]
#[allow(unused_mut)]
fn stop_python_sidecar(state: State<ApiProcess>) -> Result<(), String> {
    let mut guard = state.0.lock().unwrap();
    if let Some(mut child) = guard.take() {
        match child.kill() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to kill sidecar: {}", e)),
        }
    } else {
        Err("No sidecar running".into())
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(ApiState::default())
        .manage(ApiProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![get_api_info, stop_python_sidecar])
        .setup(|app| {
            let app_handle = app.handle().clone();

            async_runtime::spawn(async move {
                let api_state: State<ApiState> = app_handle.state();
                if let Err(e) = start_python_sidecar(app_handle.clone(), api_state).await {
                    let err_msg = format!("Failed to start Python sidecar: {}", e);
                    eprintln!("{}", err_msg);
                    let _ = app_handle.emit("sidecar-error", err_msg);
                } else {
                    println!("Python sidecar started successfully");
                    let _ = app_handle.emit("sidecar-ready", "API is ready");
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
