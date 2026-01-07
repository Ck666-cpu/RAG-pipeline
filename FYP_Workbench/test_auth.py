# FYP_Workbench/test_persistence.py
import sys
import os
import time

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from view_model import ChatViewModel


def print_separator(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def run_persistence_test():
    TEST_USER = "test_memory_user"
    TEST_PASS = "password123"

    print_separator("PHASE 1: INITIAL CHAT SESSION")

    # 1. Start App
    vm1 = ChatViewModel()

    # 2. Register/Login a fresh user for this test
    print(f" [UI] Registering/Logging in as '{TEST_USER}'...")

    # Try to login as Master first
    if not vm1.login("master", "123"):
        print(" [UI] ⚠️ Master Login Failed! resetting DB might be needed.")
        # Attempt to continue, but it will likely fail below if user doesn't exist
    else:
        # Only register if Master login succeeded
        response = vm1.register_user( TEST_USER, TEST_PASS, "Staff")
        print(f" [UI] Registration Status: {response}")
        vm1.logout()

    # 3. Login as the Test User
    if vm1.login(TEST_USER, TEST_PASS):
        print(f" [UI] Logged in successfully.")
    else:
        print(" [UI] ❌ Login failed. Verify 'users_db.json' has the user.")
        return

    # 4. Give the bot a specific fact to remember
    fact = "My secret code is OMEGA-99."
    print(f" [UI] User sending fact: '{fact}'")
    vm1.send_message(fact)

    # 5. Verify it's in memory currently
    print(f" [UI] Current History Length: {len(vm1.chat_history)}")

    # 6. SIMULATE APP CLOSE
    print(" [UI] Closing App (Destroying ViewModel)...")
    del vm1

    # ---------------------------------------------------------

    print_separator("PHASE 2: NEW SESSION (RESTART)")

    # 1. Start App Again (Clean State)
    vm2 = ChatViewModel()

    # Verify memory is empty before login
    if len(vm2.chat_history) == 0:
        print(f" [UI] ✅ App started with empty memory.")
    else:
        print(f" [UI] ❌ Memory leaked!")

    # 2. Login as the SAME User
    print(f" [UI] Logging in again as '{TEST_USER}'...")
    vm2.login(TEST_USER, TEST_PASS)

    # 3. CHECK PERSISTENCE
    print(f" [UI] History Length after login: {len(vm2.chat_history)}")

    if len(vm2.chat_history) > 0:
        last_msg = vm2.chat_history[-2]  # The user message
        print(f" [UI] ✅ Found previous message: '{last_msg.content}'")
    else:
        print(" [UI] ❌ History failed to load.")

    # 4. TEST CONTEXT AWARENESS (RAG + Memory)
    # The bot should answer based on the retrieved history, not just the new query
    question = "What is my secret code?"
    print(f" [UI] User asking: '{question}'")

    response = vm2.send_message(question)

    print(f"\n [UI] 🤖 BOT ANSWER:\n{response.content}")

    if "OMEGA-99" in response.content:
        print("\n [SUCCESS] ✅ The bot remembered the secret code across sessions!")
    else:
        print("\n [WARNING] ⚠️ Bot might have missed the context. Check LLM rewrite logic.")


if __name__ == "__main__":
    run_persistence_test()