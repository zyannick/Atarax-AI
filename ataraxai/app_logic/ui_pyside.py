import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QLabel
from PySide6.QtCore import QObject, QThread, Signal


class MockCoreAIService:
    def process_prompt(self, prompt: str, params=None) -> str:
        import time
        print(f"CORE_AI: Received prompt. Simulating 5-second processing time...")
        time.sleep(5)
        print("CORE_AI: Processing complete.")
        user_query = "your query"
        for line in prompt.splitlines():
            if line.startswith("User:"):
                user_query = line[len("User: "):]
        return f"This is Atarax-AI's response to '{user_query}'. The processing took 5 seconds."


class Worker(QObject):
    result_ready = Signal(str)

    def __init__(self, core_ai_service, prompt, gen_params):
        super().__init__()
        self.core_ai_service = core_ai_service
        self.prompt = prompt
        self.gen_params = gen_params

    def run(self):
        try:
            result = self.core_ai_service.process_prompt(self.prompt, self.gen_params)
            self.result_ready.emit(result)
        except Exception as e:
            self.result_ready.emit(f"[ERROR] An error occurred: {e}")


class AtaraxMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atarax-AI v0.1 - PySide6 UI")
        self.setGeometry(100, 100, 600, 400) 


        self.core_ai_service = MockCoreAIService()
        print("CoreAIService has been initialized and will persist.")

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        
        self.submit_button = QPushButton("Send Prompt")
        
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)

        layout = QVBoxLayout(self) 
        input_layout = QHBoxLayout()

        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)
        layout.addWidget(QLabel("Response:"))
        layout.addWidget(self.response_display)

        self.submit_button.clicked.connect(self.on_submit)
        self.prompt_input.returnPressed.connect(self.on_submit)

        self.thread = None
        self.worker = None

    def on_submit(self):
        user_query = self.prompt_input.text()
        if not user_query:
            return

        self.submit_button.setEnabled(False)
        self.response_display.setPlainText("Atarax-AI is thinking...")

        self.thread = QThread()
        full_prompt = f"User: {user_query}\nAssistant:"
        self.worker = Worker(self.core_ai_service, full_prompt, gen_params=None)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.result_ready.connect(self.thread.quit)
        self.worker.result_ready.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_result_ready(self, result: str):
        self.response_display.setPlainText(result)
        self.submit_button.setEnabled(True) 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AtaraxMainWindow()
    window.show()
    sys.exit(app.exec())