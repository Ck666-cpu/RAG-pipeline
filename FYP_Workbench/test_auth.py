# FYP_Workbench/test_streaming.py
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from view_model import ChatViewModel


def stream_print(generator, prefix="Bot: "):
    """
    Robust stream printer that works in IDE consoles.
    Prints only NEW characters as they arrive.
    """
    print(f"\n{prefix}", end="", flush=True)

    last_printed_len = 0
    start_time = time.time()

    for msg in generator:
        if msg.role == "user":
            continue

            # Get the current full content
        current_content = msg.content

        # Calculate only the NEW part we haven't printed yet
        if len(current_content) > last_printed_len:
            new_chunk = current_content[last_printed_len:]
            sys.stdout.write(new_chunk)
            sys.stdout.flush()
            last_printed_len = len(current_content)

        # Optional: Print metadata if it just appeared
        if msg.debug_sources and last_printed_len == 0:
            print(f" [Sources: {len(msg.debug_sources)} | Conf: {msg.confidence:.2f}] ", end="")

    print(f"\n(Time: {time.time() - start_time:.2f}s)\n")


def run_streaming_test():
    print("--- STREAMING TEST (IDE COMPATIBLE) ---")
    vm = ChatViewModel()

    # 1. Login
    # (Ensure you deleted the corrupted json file from previous steps first!)
    vm.login("master", "123")
    try:
        vm.register_user("test_streamer", "pass", "Admin")
    except:
        pass  # User likely exists
    vm.login("test_streamer", "pass")

    # 2. Ask Question (Memory or RAG)
    question = "Hello, who are you?"
    print(f"User: {question}")
    stream = vm.send_message(question)
    stream_print(stream)

    # 3. Ask RAG Question (If file exists)
    question2 = "What documents do you have?"
    print(f"User: {question2}")
    stream2 = vm.send_message(question2)
    stream_print(stream2)


if __name__ == "__main__":
    run_streaming_test()