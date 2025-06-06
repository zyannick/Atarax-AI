


class ChatContextManager:
    """
    Manages the context for chat interactions.
    This class is responsible for maintaining the state of the conversation,
    including the history of messages and any relevant metadata.
    """

    def __init__(self):
        self.context = []

    def add_message(self, message):
        """Adds a new message to the context."""
        self.context.append(message)

    def get_context(self):
        """Returns the current context."""
        return self.context

    def clear_context(self):
        """Clears the current context."""
        self.context = []