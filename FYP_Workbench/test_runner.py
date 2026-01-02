# FYP_Workbench/test_runner.py
import ollama
from my_brain import MyFYPBot


def run_test():
    print("--- STARTING TEST ---")

    # 1. Start the bot
    bot = MyFYPBot()

    # 2. Ask a question
    user_question = "What is the price of the house?"
    role = "admin"

    # 3. Get the answer
    result = bot.answer(user_question, role)

    # 4. Print what happened
    print("\n--- TEST RESULTS ---")
    print(f"Answer:     {result.answer}")
    print(f"Sources:    {result.sources}")
    print(f"Confidence: {result.confidence}")

    # 5. Check if it fits the rules
    if hasattr(result, 'answer') and hasattr(result, 'sources'):
        print("\n[SUCCESS] The bot output is compatible with the main project!")
    else:
        print("\n[FAIL] The bot is returning the wrong data format.")


if __name__ == "__main__":
    run_test()