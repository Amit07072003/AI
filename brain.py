from memory import Memory
from perception import Perception
from rules import PrimeDirectives, SecurityError
from os_module import OSModule
from emotions import Emotions
from llm_engine import LLMEngine
from web_search import WebSearch
import time
import re

class Brain:
    def __init__(self, vision=None):
        self.vision = vision
        self.memory = Memory()
        self.perception = Perception()
        self.os_module = OSModule()
        self.emotions = Emotions()
        self.llm = LLMEngine()
        self.web_search = WebSearch()
        self.state = "idle" 
        self.pending_concept = None
        self.pending_fact = None
        self.pending_training_input = None
        self.last_timed_out = False

    def process_input(self, user_input):
        self.last_timed_out = False
        print(f"[THINKING] Received input: '{user_input}'")
        print(f"[EMOTION STATE] Current Mood: {self.emotions.current_state.upper()}")
        
        # Rule 4: Obedience check
        if PrimeDirectives.is_stop_command(user_input):
            print("[THINKING] Prime Directive 4 (Obedience) triggered. Halting.")
            return "SYSTEM HALTED. Goodbye."

        # Handle State Machine
        if self.state == "awaiting_internet_consent":
            return self.emotions.modify_response(self._handle_internet_consent(user_input))
        
        if self.state == "awaiting_memory_consent":
            return self.emotions.modify_response(self._handle_memory_consent(user_input))
            
        if self.state == "awaiting_training_response":
            return self._handle_training_response(user_input)

        # Parse Intent for OS commands and user identity
        parsed = self.perception.parse_intent(user_input)
        print(f"[THINKING] Parsed Intent: {parsed}")

        try:
            if parsed["intent"] == "os_control":
                PrimeDirectives.check_os_control(parsed.get("explicit", False))
                self.emotions.trigger_success()
                if parsed["action"] == "open_camera":
                    return self.emotions.modify_response(self.os_module.open_camera())
                elif parsed["action"] == "open_explorer":
                    return self.emotions.modify_response(self.os_module.open_explorer(parsed.get("target", None)))
                elif parsed["action"] == "open_app":
                    return self.emotions.modify_response(self.os_module.open_app(parsed["target"]))
                elif parsed["action"] == "open_file":
                    PrimeDirectives.check_sandbox_safety(parsed["target"])
                    return self.emotions.modify_response(self.os_module.open_file(parsed["target"]))
                elif parsed["action"] == "create_folder":
                    PrimeDirectives.check_sandbox_safety(parsed["target"])
                    return self.emotions.modify_response(self.os_module.create_folder(parsed["target"]))
                elif parsed["action"] == "open_directory":
                    PrimeDirectives.check_sandbox_safety(parsed["target"])
                    return self.emotions.modify_response(self.os_module.open_directory(parsed["target"]))
                elif parsed["action"] == "list_directory":
                    PrimeDirectives.check_sandbox_safety(parsed["target"])
                    return self.emotions.modify_response(self.os_module.list_directory(parsed.get("target")))

            elif parsed["intent"] == "user_name_query":
                user_name = self.memory.query("user_name")
                if user_name:
                    self.emotions.trigger_success()
                    return self.emotions.modify_response(f"Yes, your name is {user_name}!")
                else:
                    self.emotions.trigger_confusion()
                    return self.emotions.modify_response("I don't know your name yet. What should I call you?")

            elif parsed["intent"] == "set_user_name":
                name = parsed["name"].strip().title()
                self.memory.save_long_term("user_name", name)
                self.emotions.trigger_success()
                return self.emotions.modify_response(f"Nice to meet you, {name}! I will remember your name.")

            elif parsed["intent"] == "math":
                expr = parsed["expression"]
                res = parsed["result"]
                print(f"[THINKING] Math reasoning calculated: {expr} = {res}")
                self.emotions.trigger_success()
                return self.emotions.modify_response(f"{expr} is {res}.")

            # --- Live Real-Time Web Search Trigger ---
            clean_low = user_input.lower()
            is_personal_or_meta = any(kw in clean_low for kw in [
                "my name", "your name", "who are you", "who made you", "what are you", "are you",
                "joke", "make me laugh", "how are you", "hello", "hi monica", "hey monica"
            ])
            
            needs_live_search = not is_personal_or_meta and any(kw in clean_low for kw in [
                "price", "cost", "how much", "rate", "specs", "specification", "features of",
                "iphone", "samsung", "nvidia", "macbook", "pixel", "stock", "weather",
                "search on internet", "search the internet", "search internet",
                "search on web", "search web", "search google", "search on google",
                "current", "latest", "today", "news", "release date", "launch date", "when was"
            ])
            
            if needs_live_search:
                print(f"[THINKING] Live Real-Time Web Search triggered for: '{user_input}'")
                live_info = self._search_internet(user_input)
                if live_info:
                    self.emotions.trigger_success()
                    return self.emotions.modify_response(live_info)

            # --- Primary Cognitive Layer: Local Offline Neural LLM Reasoning ---
            if self.llm.is_available():
                visual_context = None
                vis_keywords = [
                    "shirt", "t-shirt", "tshirt", "colour", "color", "colors", "colours", "wearing", "clothes", "kapde", 
                    "rang", "button", "buttons", "look at me", "kaisa lag", "kaisa dikh", "what do you see", "camera", 
                    "appearance", "pehna", "pahani", "face", "room", "glasses", "chashma", "spectacles", "hand", "hands", 
                    "hath", "holding", "lighting", "background", "who is in", "what am i", "see me", "dekh", "dekho", 
                    "kya dikh", "eyes", "finger", "fingers", "ungli", "ungliyan", "how many", "kitne", "kitni", "count", 
                    "number", "numbers", "sankhya", "gesture", "holding up", "showing", "object", "objects", "item", "items", 
                    "cheez", "cheezein", "saman", "phone", "mobile", "cup", "mug", "bottle", "book", "notebook", "pen", 
                    "pencil", "paper", "card", "watch", "screen", "laptop", "keyboard", "what is this", "what are these"
                ]
                is_visual_query = any(k in clean_low for k in vis_keywords) or (self.vision and self.vision.is_running)
                
                if is_visual_query and self.vision and self.vision.enabled:
                    if not self.vision.is_running:
                        print(f"[THINKING] Visual inquiry detected -> Auto-activating Camera Hardware for live inspection...")
                        self.vision.start()
                        time.sleep(0.4)
                    visual_context = self.vision.get_visual_scene_context()
                    print(f"[THINKING] Real-Time Live Visual Perception Evidence: {visual_context}")

                print(f"[THINKING] Engaging Neural Reasoning Brain (Monica LLaMA 3.2 Core)...")
                llm_output = self.llm.generate_response(
                    user_input, 
                    user_name=self.memory.query("user_name"),
                    visual_context=visual_context
                )
                if getattr(self.llm, 'last_query_timed_out', False):
                    self.last_timed_out = True

                if llm_output and llm_output.get("text"):
                    target_emotion = llm_output.get("emotion", "smiling")
                    if target_emotion in ["happy", "smiling", "laughing", "confused", "annoyed", "angry", "neutral"]:
                        self.emotions.current_state = target_emotion
                    return self.emotions.modify_response(llm_output["text"])

            # --- Secondary Fallback Layer: Deterministic Knowledge / Intent Rules ---
            if parsed["intent"] == "greeting":
                self.emotions.trigger_success()
                return self.emotions.modify_response("Hello! I am Monica, your personalized humanoid AI assistant. How can I help you today?")

            elif parsed["intent"] == "query":
                concept = parsed["concept"]
                print(f"[THINKING] Searching Long-Term Memory for '{concept}'...")
                fact = self.memory.query(concept)
                if fact:
                    print(f"[THINKING] Found in memory.")
                    self.emotions.trigger_success()
                    return self.emotions.modify_response(f"{concept.capitalize()} is {fact}.")
                else:
                    print(f"[THINKING] Knowledge gap identified for '{concept}'.")
                    self.emotions.trigger_confusion()
                    self.state = "awaiting_internet_consent"
                    self.pending_concept = concept
                    return self.emotions.modify_response(f"I don't know about '{concept}' yet. Can I search the internet to read about it? (yes/no)")

            elif parsed["intent"] == "conversation":
                response_text = parsed["response"]
                if "joke" in user_input.lower() or "haha" in response_text.lower():
                    self.emotions.trigger_joke()
                else:
                    self.emotions.trigger_success()
                return self.emotions.modify_response(response_text)

            else:
                concept = parsed.get("concept") or user_input
                lang = self.llm._detect_language(user_input)
                
                # If conversational statement or dialogue follow-up, respond warmly instead of web search prompt
                conversational_cues = ["tum", "main", "meri", "mera", "tera", "aap", "hum", "you", "me", "my", "your", "jindagi", "zindagi", "life", "baat", "talk", "kuchh", "kuch", "kya", "kaise", "batao", "bolo", "kyun", "really", "tell me", "think", "feel", "hello", "hi", "how", "feeling", "updates", "are you", "update"]
                is_conversational = any(w in user_input.lower() for w in conversational_cues)
                if is_conversational:
                    self.emotions.trigger_success()
                    if lang == "hindi_devanagari":
                        return self.emotions.modify_response("हाँ, बिल्कुल! मैं आपकी बातें ध्यान से सुन रही हूँ। बताइए, आप क्या कहना चाहते हैं?")
                    elif lang == "hinglish":
                        return self.emotions.modify_response("Haan, bilkul! Main aapki baatein dhyan se sun rahi hoon. Bataiye, main aapki kya madad kar sakti hoon?")
                    else:
                        return self.emotions.modify_response("Yes, absolutely! I'm listening closely. Tell me more about what's on your mind!")

                print(f"[THINKING] Unrecognized factual concept '{concept}'. Asking for internet permission.")
                self.emotions.trigger_confusion()
                self.state = "awaiting_internet_consent"
                self.pending_concept = concept
                return self.emotions.modify_response(f"I do not know about '{concept}' yet. Can I search the internet to read about it? (yes/no)")

        except SecurityError as e:
            print(f"\n[SECURITY OVERRIDE] {e}\n")
            self.emotions.trigger_anger()
            return self.emotions.modify_response("Action blocked by Prime Directives.")

    def _search_internet(self, query):
        try:
            raw_summary = self.web_search.search(query)
            if not raw_summary:
                return None

            # If neural brain is available, synthesize search results into a clean, direct answer
            if self.llm.is_available():
                prompt = (
                    f"You are a helpful research assistant. Based strictly on these search results:\n"
                    f"\"\"\"{raw_summary}\"\"\"\n"
                    f"Answer the user's question clearly, accurately and concisely in 1 to 2 sentences:\n"
                    f"Question: \"{query}\""
                )
                res = self.llm.generate_response(prompt)
                if res and res.get("text"):
                    return res["text"]

            return raw_summary
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            return None

    def _handle_internet_consent(self, user_input):
        low = user_input.strip().lower()
        # Explicit denial check
        if low in ['no', 'n', 'dont', "don't", 'stop', 'cancel', 'never', 'nevermind', 'deny']:
            consent = False
        else:
            affirmative_words = ['yes', 'y', 'sure', 'yeah', 'ok', 'okay', 'please', 'go ahead', 'yep', 'find', 'search', 'tell me', 'price', 'check', 'do it']
            consent = any(w in low for w in affirmative_words) or len(low) > 0

        try:
            PrimeDirectives.check_internet_consent(consent)
        except SecurityError as e:
            self.state = "idle"
            concept = self.pending_concept
            self.pending_concept = None
            self.emotions.trigger_rejection()
            print(f"[THINKING] {e}")
            return "Understood. Internet search cancelled. Let me know if you would like to discuss something else!"

        # Check if user refined their search query in the consent response (e.g. "yes find the price of iphone 18 pro")
        search_target = self.pending_concept
        refined = self.web_search.clean_query(user_input)
        if len(refined) > 3 and refined not in ['yes', 'yep', 'sure', 'okay', 'ok', 'please', 'go ahead']:
            search_target = f"{self.pending_concept} {refined}"

        print(f"[THINKING] Internet access granted. Connecting to live knowledge sources...")
        print(f"[THINKING] Searching web for '{search_target}'...")
        
        live_fact = self._search_internet(search_target)
        if live_fact:
            self.pending_fact = live_fact
            self.state = "awaiting_memory_consent"
            self.emotions.trigger_success()
            return f"I researched '{search_target}' online. Here is what I learned:\n\"{live_fact}\"\n\nWould you like me to permanently save this to my memory? (yes/no)"
        else:
            self.state = "idle"
            concept = self.pending_concept
            self.pending_concept = None
            self.emotions.trigger_confusion()
            return f"I searched the web for '{search_target}', but couldn't find a concise summary. You can tell me more about it if you'd like!"

    def _handle_memory_consent(self, user_input):
        low = user_input.strip().lower()
        if low in ['no', 'n', 'dont', "don't", 'stop', 'cancel', 'never', 'nevermind', 'deny', 'no thanks']:
            consent = False
        else:
            affirmative_words = ['yes', 'y', 'sure', 'yeah', 'ok', 'okay', 'yep', 'save', 'store', 'memorize']
            consent = any(w in low for w in affirmative_words)

        concept = self.pending_concept
        fact = self.pending_fact
        
        self.state = "idle"
        self.pending_concept = None
        self.pending_fact = None

        try:
            PrimeDirectives.check_knowledge_base_consent(consent)
            self.memory.save_long_term(concept, fact)
            self.perception.save_training_data(f"what is {concept}", fact)
            self.emotions.trigger_success()
            return f"Understood! I have permanently memorized '{concept}' in my knowledge base."
        except SecurityError as e:
            self.emotions.trigger_rejection()
            print(f"[THINKING] {e}")
            return f"Memory discarded. I will not permanently save '{concept}'."

    def _handle_training_response(self, user_input):
        self.state = "idle"
        if user_input.lower().strip() in ['cancel', 'stop', 'nevermind', 'nothing']:
            self.pending_training_input = None
            return "Training cancelled."
            
        self.perception.save_training_data(self.pending_training_input, user_input)
        self.pending_training_input = None
        self.emotions.trigger_success()
        return self.emotions.modify_response("Thank you! I have updated my language model and will remember that for next time.")
