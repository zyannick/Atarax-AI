import ataraxai
print("Successfully imported 'ataraxai'!")
print(f"Location: {ataraxai.__file__}")

from ataraxai import core_ai_py
print("Successfully imported 'core_ai_py' from 'ataraxai'!")
service = core_ai_py.CoreAIService()
print("Successfully created CoreAIService instance!")