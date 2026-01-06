# FYP_Workbench/test_runner.py
from fyp_service import FYPService


def run_test():
    print("--- STARTING MEMORY TEST ---")
    service = FYPService()

    # Simulate a Chat History (List of strings)
    # Let's pretend we already talked about the Tenancy Agreement
    history = [
        "User: I want to know about the Tenancy Agreement.",
        "AI: I found a document regarding a residential house lease in Malaysia."
    ]

    # ASK A VAGUE QUESTION
    # Without memory, the bot won't know what "the rent" refers to.
    question = "How much is the rent?"
    role = "admin"

    print(f"\n❓ Question: {question}")

    # Pass history into the function
    result = service.answer(question, role, history=history)

    print("\n--- RESULT ---")
    print(f"Answer:     {result.answer}")
    print(f"Sources:    {result.sources}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    run_test()