"""
Interactive Command-Line Interface for Ashcombe & Rowe Estates Chatbot.
Ideal for quick local testing and evaluation walkthroughs.
"""

import sys
import uuid
from src.agent import agent
from src.config import settings


def run_cli():
    session_id = str(uuid.uuid4())
    print("\n" + "=" * 70)
    print(" ASHCOMBE & ROWE ESTATES - LEAD QUALIFICATION CHATBOT (CLI)")
    print(f" LLM Provider: {settings.LLM_PROVIDER.upper()} | Model: {settings.LLM_MODEL}")
    print(" Type 'quit' or 'exit' to end the chat.")
    print("=" * 70 + "\n")

    # Initial start greeting
    initial_greeting = agent.chat(session_id, "Hello")
    print(f"Bot: {initial_greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("\nEnding session. Thank you!\n")
                break

            response = agent.chat(session_id, user_input)
            print(f"\nBot: {response}\n")

            state = agent.get_session_state(session_id)
            if state and state.completed:
                print(f"[Session Closed - Qualification Status: {state.qualification_status.value}]")
                break

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Goodbye.")
            break


if __name__ == "__main__":
    run_cli()
