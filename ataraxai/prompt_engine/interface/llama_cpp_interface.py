import subprocess
import os

import subprocess
import os
import json
import requests   

class LlamaCPPInterface:
    def __init__(self, mode: str,
                 model_path: str = None,
                 executable_path: str = None, 
                 server_url: str = "http://127.0.0.1:8080", 
                 default_params: dict = None):

        self.mode = mode.lower()
        self.model_path = model_path
        self.executable_path = executable_path
        self.server_url = server_url
        
        if self.mode == "cli":
            if not self.executable_path:
                raise ValueError("Parameter 'executable_path' is required for CLI mode.")
            if not self.model_path:
                raise ValueError("Parameter 'model_path' is required for CLI mode.")
            if not os.path.exists(self.executable_path):
                raise FileNotFoundError(f"Llama.cpp executable not found at: {self.executable_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        elif self.mode == "server":
            if not self.server_url:
                raise ValueError("Parameter 'server_url' is required for server mode.")
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'cli' or 'server'.")


        self._base_default_params = {
            "n_predict": 256,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "stop": [], 
            "ctx_size": 2048, 
        }
        if self.mode == "cli":
            self._base_default_params["threads"] = os.cpu_count() or 4 

        if default_params:
            self._base_default_params.update(default_params)
        
        self.default_params = self._base_default_params


    def generate(self, prompt: str, specific_params: dict = None) -> str:
        current_params = self.default_params.copy()
        if specific_params:
            current_params.update(specific_params)

        if self.mode == "cli":
            return self._generate_cli(prompt, current_params)
        elif self.mode == "server":
            return self._generate_server(prompt, current_params)
        else:
            raise RuntimeError(f"Internal error: Invalid mode '{self.mode}'.")

    def _build_cli_command(self, prompt: str, params: dict) -> list:
        cmd = [
            self.executable_path,
            "-m", self.model_path,
            "--prompt", prompt,
            "-c", str(params.get("ctx_size", 2048)),
            "-n", str(params.get("n_predict", 256)),
            "--temp", str(params.get("temperature", 0.7)),
            "--top-k", str(params.get("top_k", 40)),
            "--top-p", str(params.get("top_p", 0.9)),
        ]
        if "threads" in params:
            cmd.extend(["--threads", str(params["threads"])])
        if "repeat_penalty" in params:
            cmd.extend(["--repeat-penalty", str(params["repeat_penalty"])])
        if "stop" in params and params["stop"]:
            for stop_seq in params["stop"]:
                cmd.extend(["--reverse-prompt", stop_seq]) 

        if "grammar_file" in params and params["grammar_file"]: 
            cmd.extend(["--grammar-file", params["grammar_file"]])

        return cmd

    def _generate_cli(self, prompt: str, params: dict) -> str:
        command = self._build_cli_command(prompt, params)
        
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=params.get("timeout", 300) 
            )
            
            output_text = process.stdout
            if output_text.startswith(prompt): 
                generated_content = output_text[len(prompt):]
            else:
                prompt_lines = prompt.count('\n')
                output_lines = output_text.splitlines()
                if len(output_lines) > prompt_lines:
                    generated_content = "\n".join(output_lines[prompt_lines + (1 if not prompt.endswith("\n") else 0):])
                else: 
                    generated_content = output_text


            return generated_content.strip()

        except subprocess.CalledProcessError as e:
            error_message = f"Llama.cpp (CLI) execution failed with code {e.returncode}.\n" \
                            f"Command: {' '.join(e.cmd)}\n" \
                            f"Stderr: {e.stderr}"
            print(error_message)
            raise RuntimeError(error_message) from e
        except subprocess.TimeoutExpired as e:
            error_message = f"Llama.cpp (CLI) execution timed out.\nCommand: {' '.join(command)}"
            print(error_message)
            raise TimeoutError(error_message) from e

    def _generate_server(self, prompt: str, params: dict) -> str:
        completion_url = f"{self.server_url.rstrip('/')}/completion"
        
        payload = {
            "prompt": prompt,
            "n_predict": params.get("n_predict", self.default_params.get("n_predict")),
            "temperature": params.get("temperature", self.default_params.get("temperature")),
            "top_k": params.get("top_k", self.default_params.get("top_k")),
            "top_p": params.get("top_p", self.default_params.get("top_p")),
            "stop": params.get("stop", self.default_params.get("stop")), 
            "stream": False, 
        }
        if "grammar" in params: 
            payload["grammar"] = params["grammar"]
        if "repeat_penalty" in params:
            payload["repeat_penalty"] = params["repeat_penalty"]
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = requests.post(
                completion_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=params.get("timeout", 180)
            )
            response.raise_for_status()             
            response_data = response.json()
            if "content" in response_data:
                return response_data["content"].strip()
            else:
                print(f"Warning: 'content' key not found in server response. Response: {response_data}")

                raise ValueError("Unexpected response structure from server: 'content' key missing.")

        except requests.exceptions.HTTPError as e:
            error_message = f"Llama.cpp (Server) request failed with HTTP status {e.response.status_code}.\n" \
                            f"URL: {completion_url}\nResponse: {e.response.text}"
            print(error_message)
            raise RuntimeError(error_message) from e
        except requests.exceptions.RequestException as e: 
            error_message = f"Llama.cpp (Server) request failed.\nURL: {completion_url}\nError: {e}"
            print(error_message)
            raise RuntimeError(error_message) from e
        except json.JSONDecodeError as e:
            error_message = f"Failed to decode JSON response from Llama.cpp (Server).\nURL: {completion_url}\nError: {e}\nResponse text: {response.text if 'response' in locals() else 'N/A'}"
            print(error_message)
            raise ValueError(error_message) from e
