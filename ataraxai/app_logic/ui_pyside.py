import sys
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QLabel,
    QScrollArea,
)
from PySide6.QtCore import QObject, QThread, Signal, Qt

class MockCoreAIService:
    def process_prompt(self, prompt: str, params=None) -> str:
        import time
        print(f"CORE_AI: Received prompt. Simulating 3-second processing time...")
        time.sleep(3)
        print("CORE_AI: Processing complete.")
        user_query = "your query"
        for line in prompt.splitlines():
            if line.startswith("User:"):
                user_query = line[len("User: ") :]
        return f"This is Atarax-AI's response to '{user_query}'. The processing took 3 seconds."

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

class MessageDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("font-size: 14px; font-family: Arial; border: 1px solid #ddd; border-radius: 5px; padding: 8px;")
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setTabChangesFocus(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)


class AtaraxMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atarax-AI v0.1 - PySide6 UI")
        self.setGeometry(100, 100, 700, 500)

        self.core_ai_service = MockCoreAIService()
        print("CoreAIService has been initialized and will persist.")

        self.main_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.message_history_layout = QVBoxLayout(self.scroll_widget)
        self.message_history_layout.addStretch() 
        self.scroll_area.setWidget(self.scroll_widget)
        
        self.input_bar_layout = QHBoxLayout()
        
        self.user_icon = QLabel("🧑‍💻") 
        self.user_icon.setStyleSheet("font-size: 24px;")

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.returnPressed.connect(self.on_submit)

        self.button_layout = QVBoxLayout()
        self.submit_button = QPushButton("Send")
        self.submit_button.clicked.connect(self.on_submit)

        self.voice_input_button = QPushButton("Speak")
        self.voice_input_button.clicked.connect(self.on_voice_input)

        self.button_layout.addWidget(self.submit_button)
        self.button_layout.addWidget(self.voice_input_button)

        self.input_bar_layout.addWidget(self.user_icon)
        self.input_bar_layout.addWidget(self.user_input)
        self.input_bar_layout.addLayout(self.button_layout)

        self.main_layout.addWidget(self.scroll_area) 
        self.main_layout.addLayout(self.input_bar_layout) 

        self.thread = None
        self.worker = None

    def add_message(self, text, alignment):
        message_widget = MessageDisplay()
        message_widget.setPlainText(text)
        
        line_layout = QHBoxLayout()
        if alignment == Qt.AlignLeft:
            line_layout.addWidget(message_widget, stretch=0)
            line_layout.addStretch(1)
        else: # AlignRight
            line_layout.addStretch(1)
            line_layout.addWidget(message_widget, stretch=0)
            
        self.message_history_layout.insertLayout(self.message_history_layout.count() - 1, line_layout)
        
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        
        return message_widget


    def on_voice_input(self):
        print("Voice input feature is not implemented yet.")
        self.add_message("Voice input feature is not implemented yet.", Qt.AlignLeft)

    def on_submit(self):
        # NOTE: Fixed variable name from self.prompt_input to self.user_input
        user_query = self.user_input.text().strip()
        if not user_query:
            return

        # Add user's message to the UI immediately
        self.add_message(user_query, Qt.AlignRight)
        self.user_input.clear() # Clear the input field

        # Disable button and add a "thinking" message
        self.submit_button.setEnabled(False)
        self.voice_input_button.setEnabled(False)
        thinking_message = self.add_message("Thinking...", Qt.AlignLeft)

        # Threading logic (correct as you wrote it)
        self.thread = QThread()
        full_prompt = f"User: {user_query}\nAssistant:"
        self.worker = Worker(self.core_ai_service, full_prompt, gen_params=None)
        self.worker.moveToThread(self.thread)

        # NOTE: Pass the "thinking" widget to the result slot using a lambda
        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(lambda result: self.on_result_ready(result, thinking_message))
        self.worker.result_ready.connect(self.thread.quit)
        self.worker.result_ready.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_result_ready(self, result: str, message_widget_to_update: MessageDisplay):
        message_widget_to_update.setPlainText(result)
        self.submit_button.setEnabled(True)
        self.voice_input_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AtaraxMainWindow()
    window.show()
    sys.exit(app.exec())