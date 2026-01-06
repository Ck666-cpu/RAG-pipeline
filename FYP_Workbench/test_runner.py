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
    question = "What is the date of the tenancy agreement?"
    role = "admin"

    print(f"\n❓ Question: {question}")

    # Pass history into the function
    result = service.answer(question, role, history=history)

    print("\n--- RESULT ---")
    print(f"Answer:     {result.answer}")
    print(f"Sources:    {result.sources}")
    print(f"Confidence: {result.confidence:.2f}")

    # In your code where you get the response
    response = query_engine.query("How much is the rent?")

    # ADD THIS DEBUG PRINT:
    print("\n--- DEBUG: WHAT THE LLM SAW ---")
    for node in response.source_nodes:
        print(f"Content: {node.get_content()}")
        print(f"Score: {node.score}")
    print("-------------------------------\n")

if __name__ == "__main__":
    run_test()