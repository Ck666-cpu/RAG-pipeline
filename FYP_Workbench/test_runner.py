# FYP_Workbench/test_runner.py
import time
from fyp_service import FYPService


def run_test():
    print("--- STARTING ARCHITECTURE TEST ---")

    # 1. Initialize the Service (Like the Backend would)
    service = FYPService()

    # 2. (Optional) Upload a test file if you haven't yet
    # path = input("Enter path to a PDF to test upload (or press Enter to skip): ").strip()
    # if path:
    #     success, msg = service.upload_document(path)
    #     print(f"Upload Status: {msg}")

    # 3. Ask a Question
    question = "What is this document about?"
    role = "admin"

    start = time.time()
    result = service.answer(question, role)
    duration = time.time() - start

    # 4. Validate Output
    print("\n--- RESULT FROM SERVICE ---")
    print(f"Answer:     {result.answer}")
    print(f"Sources:    {result.sources}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Time Taken: {duration:.2f}s")

    if hasattr(result, 'answer'):
        print("\n✅ [SUCCESS] Service returns valid Data Object!")
    else:
        print("\n❌ [FAIL] Service returned wrong format.")


if __name__ == "__main__":
    run_test()