from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
import sys

queries = ["Scrie o scurtă poezie.", "Imaginați-vă că participați la o cursă cu un grup de oameni. Dacă tocmai ai depășit-o pe a doua persoană, care este poziția ta actuală? Unde este persoana pe care tocmai ai depășit-o?",
            "Imaginează-te ca un copac de 100 de ani într-o pădure luxuriantă, care se ocupă de treburile tale, când dintr-o dată, o grămadă de despăduritori apare să te taie. Cum te simți când tipii ăia încep să te atace?",
            "Două trenuri, trenul A și trenul B, circulă unul spre celălalt. Încep la 10 km unul de celălalt. Trenul A circulă cu 10 m/s, iar trenul B cu 5 km/h. O insectă zboară înainte și înapoi între ambele trenuri cu 50 km/h. La ce distanță va zbura insecta înainte ca ambele trenuri să se atingă?",
            "Cum îmi pot securiza routerul wifi de acasă?", "De ce echipament ai nevoie să faci scuba diving?", "Care este rezultatul următorului calcul: 2x4 + 3?",
            "Ce jocuri de societate pot juca cu prietenii mei?",
            ]


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
