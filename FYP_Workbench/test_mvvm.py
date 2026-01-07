# FYP_Workbench/test_mvvm.py
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from view_model import ChatViewModel


def print_separator(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)


def simulate_frontend_interaction():
    # 1. INITIALIZE VIEWMODEL (The "Binding" step)
    print(" [UI] Initializing App...")
    vm = ChatViewModel()

    # ---------------------------------------------------------
    # TEST SCENARIO 1: ADMIN USER (Technical Response)
    # ---------------------------------------------------------
    print_separator("SCENARIO 1: ADMIN ROLE")

    # Simulate User selecting "Admin" from dropdown
    print(" [UI] User changed role to 'Admin'")
    vm.set_user_role("Admin")

    # Simulate User clicking "Send"
    question = "How to write a tenancy agreement?"
    print(f" [UI] User typed: '{question}'")

    # Check status (optional, just like a loading spinner)
    print(f" [UI] Status: {vm.status_message}")

    # Send message
    response = vm.send_message(question)

    # Display result
    print(f" [UI] Bot Answer: {response.content}")
    print(f" [UI] Confidence: {response.confidence:.2f}")

    # ---------------------------------------------------------
    # TEST SCENARIO 2: TENANT USER (Friendly Response)
    # ---------------------------------------------------------
    print_separator("SCENARIO 2: TENANT ROLE")

    # Simulate User selecting "Tenant" from dropdown
    print(" [UI] User changed role to 'User'")
    vm.set_user_role("User")

    # Simulate User asking the SAME question
    print(f" [UI] User typed: '{question}'")

    response = vm.send_message(question)

    print(f" [UI] Bot Answer: {response.content}")
    print(f" [UI] Confidence: {response.confidence:.2f}")

    # ---------------------------------------------------------
    # TEST SCENARIO 3: UPLOAD FILE
    # ---------------------------------------------------------
    print_separator("SCENARIO 3: FILE UPLOAD")

    # Replace this with a real path on your machine for testing
    test_file = "test_document.txt"

    if os.path.exists(test_file):
        print(f" [UI] Uploading {test_file}...")
        result_msg = vm.upload_document(test_file)
        print(f" [UI] Upload Result: {result_msg}")
    else:
        print(" [UI] Skipping upload test (create 'test_document.txt' to test this).")


if __name__ == "__main__":
    simulate_frontend_interaction()