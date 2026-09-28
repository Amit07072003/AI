import json
import os

class Memory:
    def __init__(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_term_memory.json")
        self.filepath = filepath
        self.working_memory = {}
        self.knowledge_gaps = []
        self._initialize_long_term_memory()

    def _initialize_long_term_memory(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({"facts": {}}, f, indent=4)

    def load_long_term(self):
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def save_long_term(self, concept, fact):
        data = self.load_long_term()
        data["facts"][concept] = fact
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[THINKING] Memory saved: {concept} -> {fact}")

    def query(self, concept):
        data = self.load_long_term()
        return data["facts"].get(concept, None)
