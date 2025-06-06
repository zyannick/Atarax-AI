


class ChatMemory:
    """
    A class to manage chat memory, allowing for the addition and retrieval of messages.
    """

    def __init__(self):
        self.messages = []

    def add_message(self, message: str):
        """
        Adds a message to the chat memory.

        :param message: The message to be added.
        """
        self.messages.append(message)

    def get_messages(self) -> list:
        """
        Retrieves all messages from the chat memory.

        :return: A list of messages.
        """
        return self.messages