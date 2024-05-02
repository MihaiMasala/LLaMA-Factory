from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import sys

queries = ["Scrie o scurtă poezie.", "Cât fac 2 + 3?", "Numele meu este Mihai și sunt "]

try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")


def main():
    query_id = 0

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        if query_id >= 2*len(queries):
            sys.exit()
        try:
            if query_id % 2 == 0:
                query = queries[query_id//2]
                
            else:
                query = "clear"
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise
        query_id += 1

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            # print("History has been removed.")
            continue

        print("User:", query)
        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)
        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        print("#"*80)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
