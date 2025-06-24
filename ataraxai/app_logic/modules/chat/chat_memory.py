


class ChatMemory:


    def __init__(self):
        self.messages = []

    def add_message(self, message: str):
        self.messages.append(message)

    def get_messages(self) -> list:
        return self.messages