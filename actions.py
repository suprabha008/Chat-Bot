from transformers import pipeline
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher

# Load Hugging Face model once
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

class ActionChatWithAI(Action):
    def name(self):
        return "chat_with_ai"  # ✅ Must match domain.yml and rules.yml

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text", "")

        if not user_message:
            dispatcher.utter_message(text="Sorry, I didn't understand. Can you repeat?")
            return []

        try:
            # Generate response
            response = chatbot(user_message, max_length=50, num_return_sequences=1, truncation=True)

            # Extract the generated text
            bot_reply = response[0].get("generated_text", "").strip()

            # Prevent repeating user input
            if bot_reply.lower().startswith(user_message.lower()):
                bot_reply = bot_reply[len(user_message):].strip()

            # If response is empty, provide a fallback
            if not bot_reply:
                bot_reply = "I'm still learning! Can you ask me something else?"

        except Exception as e:
            bot_reply = "Oops! Something went wrong while generating a response."

        dispatcher.utter_message(text=bot_reply)
        return []
