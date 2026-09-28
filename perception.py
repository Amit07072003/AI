import re
import json
import os
import difflib

class Perception:
    def __init__(self, training_file=None):
        if training_file is None:
            training_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.json")
        self.training_file = training_file
        self.training_data = []
        self._load_training_data()

    def _load_training_data(self):
        if not os.path.exists(self.training_file):
            with open(self.training_file, 'w') as f:
                json.dump({"conversations": []}, f, indent=4)
        with open(self.training_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.training_data = data.get("conversations", [])

    def save_training_data(self, user_input, ai_response):
        self.training_data.append({"input": user_input, "response": ai_response})
        with open(self.training_file, 'w', encoding='utf-8') as f:
            json.dump({"conversations": self.training_data}, f, indent=4)
        print(f"[THINKING] Appended to training data: '{user_input}' -> '{ai_response}'")

    def _clean_tokens(self, text):
        cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
        return [w for w in cleaned.split() if w]

    def _calculate_similarity(self, query_tokens, train_tokens):
        if not query_tokens or not train_tokens:
            return 0.0

        STOPWORDS = {
            "who", "what", "where", "when", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "of", "in", "on", "at", "by", "for", "with", "about",
            "to", "from", "do", "does", "did", "can", "could", "would", "should",
            "me", "my", "your", "you", "it", "its", "tell", "just", "ans", "answer"
        }

        # High-value semantic keywords
        KEYWORD_WEIGHTS = {
            "capabilities": 3.0, "capability": 3.0, "features": 3.0, "skills": 3.0,
            "speak": 2.5, "speaking": 2.5, "voice": 2.5, "talk": 2.5,
            "hear": 2.5, "listening": 2.5, "listen": 2.5,
            "see": 2.5, "camera": 2.5, "vision": 2.5, "look": 2.0,
            "name": 3.0, "creator": 3.0, "created": 3.0, "made": 3.0, "architect": 3.0,
            "joke": 3.0, "funny": 2.5, "laugh": 2.5,
            "emotions": 3.0, "emotion": 3.0, "feel": 2.5, "feeling": 2.5,
            "directives": 3.0, "rules": 3.0, "prime": 3.0,
            "operate": 3.0, "control": 3.0, "laptop": 3.0, "pc": 3.0, "computer": 3.0,
            "help": 2.5, "sentient": 2.5, "llm": 2.5,
            "beautiful": 2.5, "gorgeous": 2.5, "awesome": 2.5,
            "goodbye": 2.5, "bye": 2.5,
            "aata": 3.0, "aati": 3.0, "kaise": 2.5, "kaisi": 2.5, "naam": 3.0,
            "madad": 2.5, "shukriya": 2.5, "dhanyawad": 2.5
        }

        def get_weight(w):
            if w in KEYWORD_WEIGHTS:
                return KEYWORD_WEIGHTS[w]
            elif w in STOPWORDS:
                return 0.15  # Filler / stop words carry very low weight
            else:
                return 2.0   # Distinct nouns / entity keywords carry high weight

        # Fuzzy word matching
        matched_weight = 0.0
        total_weight = 0.0

        for q in query_tokens:
            q_wt = get_weight(q)
            total_weight += q_wt
            
            # Find best match in train_tokens
            best_word_sim = 0.0
            for t in train_tokens:
                if q == t:
                    best_word_sim = 1.0
                    break
                else:
                    ratio = difflib.SequenceMatcher(None, q, t).ratio()
                    if ratio > 0.85 and ratio > best_word_sim:
                        best_word_sim = ratio
            
            matched_weight += q_wt * best_word_sim

        # Pronoun conflict check (my vs your / me vs you)
        if ("my" in query_tokens and "your" in train_tokens) or ("your" in query_tokens and "my" in train_tokens):
            return 0.0
        if ("me" in query_tokens and "you" in train_tokens) or ("you" in query_tokens and "me" in train_tokens):
            # Allow unless they are short 2-3 word queries
            if len(query_tokens) <= 4:
                return 0.0

        train_weight = sum(get_weight(t) for t in train_tokens)
        
        recall = matched_weight / total_weight if total_weight > 0 else 0
        precision = matched_weight / train_weight if train_weight > 0 else 0
        
        # F1-style harmonic blend
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        return 0.0

    def _check_math(self, text):
        clean = text.lower().strip().rstrip('=?').strip()
        for prefix in ["what is ", "calculate ", "compute ", "solve ", "how much is ", "what is the value of "]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        
        words_map = {
            "plus": "+", "add": "+",
            "minus": "-", "subtract": "-",
            "times": "*", "multiplied by": "*", "into": "*",
            "divided by": "/", "divide by": "/", "over": "/"
        }
        expr = clean
        for w, op in words_map.items():
            expr = re.sub(r'\b' + re.escape(w) + r'\b', op, expr)
        
        # Check if expr looks like math
        if re.match(r'^[\d\s\+\-\*\/\(\)\.\%\^x]+$', expr) and any(c.isdigit() for c in expr):
            try:
                safe_expr = expr.replace('^', '**').replace('x', '*')
                val = eval(safe_expr, {'__builtins__': None}, {})
                if isinstance(val, (int, float)):
                    if isinstance(val, float) and val.is_integer():
                        val = int(val)
                    elif isinstance(val, float):
                        val = round(val, 4)
                    return {"intent": "math", "expression": clean, "result": str(val), "explicit": False}
            except ZeroDivisionError:
                return {"intent": "math", "expression": clean, "result": "undefined (division by zero is impossible)", "explicit": False}
            except Exception:
                pass
        return None

    def parse_intent(self, text):
        clean_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        # 1. Check Mathematical and Calculation Expressions
        math_result = self._check_math(text)
        if math_result:
            return math_result

        # 2. Check Hardcoded OS commands first (Highest Priority)
        # Check Directory Content Listing (e.g. "tell me all content available in it", "list files", "show directory")
        list_triggers = [
            "tell me all content available in it", "tell me all content in it", "tell me content in it",
            "tell me all files in it", "show all content", "show content", "list files", "list directory",
            "show files", "what files are in it", "what is inside it", "what is in this folder", "what is in d drive",
            "list all files", "show all files", "list content"
        ]
        if clean_text in list_triggers or clean_text.startswith("list files in ") or clean_text.startswith("show files in "):
            target_dir = clean_text.replace("list files in ", "").replace("show files in ", "").strip()
            if target_dir in list_triggers: target_dir = "it"
            return {"intent": "os_control", "action": "list_directory", "target": target_dir if target_dir else "it", "explicit": True}

        # Check Drive Opening (e.g. "open d drive", "open d derive", "open c drive", "open d:")
        drive_match = re.match(r'^open\s+([a-zA-Z])(?:\s*(?:drive|derive|disk|partition|:))?$', clean_text)
        if drive_match:
            drive_letter = drive_match.group(1).upper()
            return {"intent": "os_control", "action": "open_directory", "target": f"{drive_letter}:\\", "explicit": True}

        if clean_text in ["open file explorer", "open explorer", "open my computer", "launch explorer", "launch file explorer", "open file manager", "open files"]:
            return {"intent": "os_control", "action": "open_explorer", "target": None, "explicit": True}
        if clean_text in ["open notepad", "launch notepad"]:
            return {"intent": "os_control", "action": "open_app", "target": "notepad", "explicit": True}
        if clean_text in ["open calculator", "open calc", "launch calculator"]:
            return {"intent": "os_control", "action": "open_app", "target": "calc", "explicit": True}
        if clean_text.startswith("open camera"):
            return {"intent": "os_control", "action": "open_camera", "explicit": True}
        if clean_text.startswith("open file "):
            filename = clean_text.replace("open file ", "").strip()
            if filename in ["explorer", "file explorer", "manager"]:
                return {"intent": "os_control", "action": "open_explorer", "target": None, "explicit": True}
            return {"intent": "os_control", "action": "open_file", "target": filename, "explicit": True}
        if clean_text.startswith("create folder ") or clean_text.startswith("make folder "):
            folder_name = clean_text.replace("create folder ", "").replace("make folder ", "").strip()
            return {"intent": "os_control", "action": "create_folder", "target": folder_name, "explicit": True}
        if clean_text.startswith("open directory ") or clean_text.startswith("open folder ") or clean_text.startswith("open drive "):
            dirname = clean_text.replace("open directory ", "").replace("open folder ", "").replace("open drive ", "").strip()
            if not dirname or dirname in ["explorer", "file explorer", "here", "this", "files"]:
                return {"intent": "os_control", "action": "open_explorer", "target": None, "explicit": True}
            return {"intent": "os_control", "action": "open_directory", "target": dirname, "explicit": True}

        # 3. Check User Identity Queries
        if clean_text in ["what is my name", "do you know my name", "who am i", "do you remember my name", "tell me my name", "do you know who i am"]:
            return {"intent": "user_name_query", "explicit": False}

        for prefix in ["my name is ", "call me ", "i am called "]:
            if clean_text.startswith(prefix):
                name = clean_text[len(prefix):].strip()
                if name and name not in ["good", "fine", "sad", "happy", "ok", "okay", "tired", "back", "here", "ready"]:
                    return {"intent": "set_user_name", "name": name, "explicit": False}

        # 4. Check AI Self-Identity queries (Monica / Monika / You)
        ai_self_keywords = [
            "who is monica", "who is monika", "who are you", "what is your name",
            "do you know who is monica", "do you know who is monika", "do you know monica", "do you know monika",
            "are you monica", "are you monika", "what is monica", "what is monika", "tell me about yourself",
            "tell me about monica", "tell me about monika"
        ]
        if clean_text in ai_self_keywords:
            return {
                "intent": "conversation",
                "response": "That's me! I am Monica, your personalized humanoid AI cognitive brain.",
                "score": 1.0,
                "explicit": False
            }
            
        # 5. Check Trainable Similarity
        self._load_training_data()
        query_tokens = self._clean_tokens(text)
        
        best_match = None
        best_score = 0.0
        
        for convo in self.training_data:
            train_tokens = self._clean_tokens(convo["input"])
            score = self._calculate_similarity(query_tokens, train_tokens)
            if score > best_score:
                best_score = score
                best_match = convo

        # 4. Check simple greetings
        if clean_text in ["hello", "hi", "hey", "hola"]:
            return {"intent": "greeting", "explicit": False}

        # 5. Check Knowledge Query prefixes (What is, Who is, Tell me about, etc.)
        prefixes = [
            "what is ", "who is ", "what are ", "tell me about ", 
            "explain ", "search for ", "do you know about ", "do you know "
        ]
        for prefix in prefixes:
            if clean_text.startswith(prefix):
                concept = clean_text[len(prefix):].strip()
                # If there is a high-confidence match (e.g., 'what is your name', 'who are you', 'who made you')
                if best_score >= 0.75 and best_match:
                    return {"intent": "conversation", "response": best_match["response"], "score": best_score, "explicit": False}
                if concept:
                    if concept in ["monica", "monika", "who is monica", "who is monika", "yourself", "who are you"]:
                        return {
                            "intent": "conversation",
                            "response": "That's me! I am Monica, your personalized humanoid AI cognitive brain.",
                            "score": 1.0,
                            "explicit": False
                        }
                    return {"intent": "query", "concept": concept, "explicit": False}

        # 6. If confidence is strong, return trained conversation response
        if best_score >= 0.55 and best_match:
            return {"intent": "conversation", "response": best_match["response"], "score": best_score, "explicit": False}
            
        # 7. Unknown query / concept
        return {"intent": "unknown", "concept": clean_text, "raw": text, "explicit": False}
