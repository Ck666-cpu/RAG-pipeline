# FYP_Workbench/test_auth.py
from view_model import ChatViewModel


def test_permissions():
    vm = ChatViewModel()

    print("--- 1. TEST LOGIN (Master Admin) ---")
    vm.login("master", "123")  # Default created by user_manager
    print(f"Logged in as: {vm.current_user.role}")

    print("\n--- 2. TEST MASTER ADMIN RESTRICTIONS ---")
    # Master Admin should FAIL to chat
    response = vm.send_message("Hello?")
    print(f"Chat Attempt: {response.content}")
    # Expected: "Master Admins do not have access..."

    print("\n--- 3. TEST USER MANAGEMENT ---")
    # Master Admin creates a Staff and an Admin
    print(vm.register_user("john_staff", "pass", "Staff"))
    print(vm.register_user("jane_admin", "pass", "Admin"))

    print("\n--- 4. TEST STAFF ACTIONS ---")
    vm.login("john_staff", "pass")
    print(f"Logged in as: {vm.current_user.role}")

    # Staff tries to upload GLOBAL doc (Should Fail)
    print(f"Upload Global Attempt: {vm.upload_document('dummy.txt', is_global=True)}")
    # Expected: "Only Admins can upload Global documents."

    print("\n--- 5. TEST ADMIN ACTIONS ---")
    vm.login("jane_admin", "pass")

    # Admin upload GLOBAL doc (Should Succeed)
    # (Note: This will return 'File path not found' if dummy.txt doesn't exist,
    # but the Permission Check passed!)
    print(f"Upload Global Attempt: {vm.upload_document('dummy.txt', is_global=True)}")


if __name__ == "__main__":
    test_permissions()