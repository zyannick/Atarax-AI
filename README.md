<img src="https://github.com/user-attachments/assets/fdd8be29-ac97-4efc-8e5b-b559096e5234" alt="AtaraxAI" width="200">


# Atarax-AI

A Local, Privacy-Preserving AI Assistant Powered by llama.cpp

_Fully offline. Multi-modal. Secure. Yours._

Develop a full-featured AI assistant that runs entirely offline using llama.cpp, optimized for low-latency, high-accuracy inference on consumer-grade hardware (laptop, smartphone)

The assistant supports multi-modal inputs (text + voice + images/videos), performs real-time reasoning, and integrates with local system APIs (calendar, file system, etc.)  all with zero cloud dependency.

# Key Features

- Quantized Model Deployment: Use 4-bit or 5-bit quantization with llama.cpp to make large models run smoothly on CPU or edge devices.
- Custom Prompt Engineering Engine: Create a modular prompting system (like reusable “prompt chains”) to allow dynamic behaviors (e.g., “summarize recent emails,” “generate shell scripts,” “answer from local docs”).
- Voice Interface: Integrate Whisper.cpp for speech-to-text, making it a voice-capable offline assistant.
- Local Contextual Memory: Maintain persistent local memory (e.g., SQLite + embeddings) for personal assistant functionality.
- Privacy-First Design: No cloud calls; all processing and data storage is local. Emphasize encryption, transparency, and control.
- Smart Caching and Context Window Management: Implement sliding-window techniques, smart summarization, and embeddings to keep long-term context in RAM-limited environments.

# Planning
  - [x] Core Setup with llama.cpp
  - [ ] Automatic local benchmarking
      - In progress
  - [ ] Frompt engineering framework
  - [ ] Embedding Store + Context Management
  - [ ] System Integration
  - [ ] Whisper.cpp Integration
  - [ ] Vision model integration
  - [ ] UI Layer
  - [ ] Testing, Benchmarking, Optimization 


# Modules

## Benchmarker

We try to setup a module to automatically benchmark the models on the user laptop and them select the best model and the best quantization. 

- First, we will automatically parse and download a list of models from internet. For each model we will try to benchmark for the parameters  (standard prompt, perplexity dataset path, number of tokens to generate, threads, context size)

- For each model we will first make a warm-up run to ensure that the measurements are stable as caches are warmed up

