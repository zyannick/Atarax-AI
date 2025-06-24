


class ChatContextManager:


    def __init__(self):
        self.context = []

    def add_message(self, message):
        self.context.append(message)

    def get_context(self):
        return self.context

    def clear_context(self):
        self.context = []