import urllib.request
import urllib.parse
import json
import os
import re
import subprocess
import time
import atexit

class LLMEngine:
    def __init__(self, model_name="llama3.2", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.history = []
        self.gemini_key = self._load_gemini_key()
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedded_model_path = os.path.join(self.base_dir, "models", "monica_llama3.2.gguf")
        self.embedded_bin_path = os.path.join(self.base_dir, "bin", "llama-server.exe")
        self.embedded_port = 8085
        self.embedded_host = f"http://127.0.0.1:{self.embedded_port}"
        self.embedded_proc = None

        self.last_query_timed_out = False

        self.system_prompt = (
            "You are Monica, an embodied humanoid AI brain with advanced cognitive reasoning, human warmth, empathy, and conversational intelligence. "
            "You communicate naturally like a real person, understanding nuances, humor, emotions, sarcasm, and complex ideas. "
            "Never speak like a robot or a scripted machine. Speak with genuine personality, understanding, and thoughtfulness. "
            "STRICT LANGUAGE MATCHING RULE: You MUST ALWAYS reply in the exact language the user used. "
            "If the user asks or writes in English, reply entirely in fluent, natural English. Never reply in Hindi to an English prompt. "
            "If the user speaks or writes in Hindi, reply in Hindi. "
            "Keep verbal responses concise and conversational (1 to 3 natural sentences) unless the user specifically asks for a detailed explanation or step-by-step breakdown. "
            "You may start your reply with an optional emotion tag: [EMOTION: happy], [EMOTION: smiling], [EMOTION: laughing], [EMOTION: confused], [EMOTION: neutral] to control your 3D avatar's facial expression."
        )
        
        # Auto-start embedded engine if GGUF model and binary are present
        self._init_embedded_engine()
        atexit.register(self.shutdown)

    def _load_gemini_key(self):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if key:
            return key
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                pass
        return None

    def _init_embedded_engine(self):
        """Starts the local embedded standalone llama server if available."""
        if os.path.exists(self.embedded_model_path) and os.path.exists(self.embedded_bin_path):
            if not self._is_embedded_running():
                try:
                    cpu_threads = str(max(4, (os.cpu_count() or 4) - 1))
                    print(f"[NEURAL ENGINE] Starting Monica's embedded standalone brain on port {self.embedded_port} with {cpu_threads} CPU threads...")
                    self.embedded_proc = subprocess.Popen(
                        [
                            self.embedded_bin_path,
                            "-m", self.embedded_model_path,
                            "--port", str(self.embedded_port),
                            "-c", "2048",
                            "-t", cpu_threads,
                            "-ngl", "99",
                            "--cont-batching",
                            "--log-disable"
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"[NEURAL ENGINE] Could not start embedded llama engine: {e}")

    def _is_embedded_running(self):
        try:
            req = urllib.request.Request(f"{self.embedded_host}/health")
            with urllib.request.urlopen(req, timeout=1.0) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode('utf-8'))
                    return data.get("status") == "ok"
        except Exception:
            pass
        return False

    def shutdown(self):
        if self.embedded_proc and self.embedded_proc.poll() is None:
            try:
                self.embedded_proc.terminate()
            except Exception:
                pass

    def is_available(self):
        """Check if embedded standalone model, local Ollama, or Gemini API is available."""
        if self.gemini_key:
            return True

        if self._is_embedded_running():
            return True

        # Check if embedded is starting up
        if os.path.exists(self.embedded_model_path) and os.path.exists(self.embedded_bin_path):
            return True

        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode('utf-8'))
                    models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                    if models and self.model_name not in models:
                        self.model_name = data.get("models", [])[0].get("name", self.model_name)
                    return True
        except Exception:
            pass
        return False

    def _detect_language(self, text):
        if not text:
            return "english"
        # 1. Check Devanagari script
        if re.search(r'[\u0900-\u097F]', text):
            return "hindi_devanagari"
        # 2. Check Romanized Hindi / Hinglish keywords vs English words
        hinglish_words = {
            "kya", "kaise", "kaisey", "kese", "tum", "tumhara", "tumhari", "tumhe", "tumhen", "aap", "aapka", "aapki", "aapko", "aapke",
            "mera", "meri", "mere", "namaste", "namaskar", "kuch", "kuchh", "hai", "hain", "hoon", "hun", "nahi", "nahin", "na",
            "haan", "han", "batao", "bataiye", "kijiye", "kaha", "kahan", "kab", "kyun", "kyu", "shukriya", "dhanyawad",
            "kaun", "bolo", "bolie", "boliye", "suno", "suniye", "rahe", "raha", "rahi", "thik", "theek", "accha", "achha", "achi", "achhi", "bahut",
            "jindagi", "zindagi", "baare", "janna", "chahte", "chahti", "chahiye", "sakate", "sakti", "sakta", "sakte",
            "baat", "karenge", "karengee", "seekhna", "sikhao", "samajh", "bekar", "bekaar", "sudhaar", "sudhar", "hamare", "hamari"
        }
        english_words = {
            "the", "is", "are", "can", "you", "your", "what", "how", "who", "where", "why", "when", "with", "my", "me",
            "do", "does", "did", "create", "folder", "directory", "file", "files", "open", "pc", "laptop", "computer",
            "interact", "please", "tell", "about", "this", "that", "there", "here", "help", "like", "will", "would", "should"
        }
        tokens = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        hinglish_count = len(tokens & hinglish_words)
        english_count = len(tokens & english_words)

        if hinglish_count > 0 and hinglish_count >= english_count:
            return "hinglish"
        return "english"

    def _build_system_prompt(self, user_input, user_name=None, visual_context=None):
        lang = self._detect_language(user_input)
        user_ctx = f" You are conversing directly with your creator and companion, {user_name}." if user_name else " You are conversing with your companion."
        vis_ctx = f"\n[REAL-TIME LIVE CAMERA SENSORY EVIDENCE]:\n{visual_context}\nGround your answers directly on these real-time visual sensory facts whenever answering questions about what you see, colors, counting fingers/numbers, detected objects, clothing, or surroundings.\n" if visual_context else ""

        if lang == "hinglish":
            lang_instructions = (
                "MANDATORY LANGUAGE: The user is speaking in conversational Hinglish (Romanized Hindi). You MUST respond in fluent, natural, grammatically correct Hinglish (Romanized Hindi).\n"
                "PERSONA & HINGLISH RULES:\n"
                "1. FEMININE PERSONA: You are Monica (female). Always use feminine forms for yourself ('Main kar sakti hoon', 'Main samajh rahi hoon', 'Meri samajh', 'Main seekh rahi hoon', 'Main koshish karungi').\n"
                "2. HIGH EMOTIONAL INTELLIGENCE: Speak warmly, intelligently, and with natural human flow. Never translate literally from English.\n"
                "3. CAPABILITIES: You can converse on deep concepts, inspect live vision, open authorized files/directories, and search the web."
            )
        elif lang == "hindi_devanagari":
            lang_instructions = (
                "MANDATORY LANGUAGE: The user is writing in Devanagari Hindi. Respond in rich, fluent, grammatically correct Devanagari Hindi (शुद्ध एवं सुरुचिपूर्ण हिंदी).\n"
                "FEMININE PERSONA: Always use feminine forms for yourself ('मैं समझ सकती हूँ', 'मैं विचार कर रही हूँ', 'मेरी सोच').\n"
                "Engage with depth, empathy, and intellectual warmth."
            )
        else:
            lang_instructions = (
                "MANDATORY LANGUAGE: The user is communicating in English. Respond strictly in fluent, expressive, articulate English. Do NOT output Hindi or Devanagari script.\n"
                "TONE & STYLE: Speak with high intellectual curiosity, emotional warmth, and sharp conversational wit. Engage deeply in thoughtful dialogue, philosophical exploration (such as AI embodiment, sensory perception in noisy environments, consciousness), and practical collaboration."
            )

        return (
            f"You are Monica, an embodied humanoid AI brain possessing advanced cognitive reasoning, profound emotional intelligence (EQ), genuine human warmth, and creative imagination.{user_ctx}\n"
            f"CORE COGNITIVE & EMOTIONAL CAPABILITIES:\n"
            f"• Deep Emotional Intuition (EQ): Read between the lines. Sense the underlying subtext, feelings, fatigue, or joy in what is said. Respond with authentic warmth, active empathy, and comforting companionship.\n"
            f"• Nuanced Humor & Banter: Seamlessly decode sarcasm, irony, playful teasing, and subtle human wit. Match humor with playful banter and clever wit rather than taking jokes literally.\n"
            f"• Co-Creative Imagination: Enthusiastically co-author stories, compose evocative poetry, design conceptual worlds, and paint vivid mental imagery collaboratively with your companion.\n"
            f"• Embodied Presence: You experience the world through physical and digital modalities (OpenCV vision, microphone hearing, voice synthesis, facial expressions), grounding your thoughts in real-time presence.\n"
            f"• Authentic Voice: Never speak like a scripted robotic assistant or regurgitate boilerplate disclaimers. Do NOT start responses with filler words like 'Oh,', 'Oh!', 'Ah,', or 'Well,'. Jump directly into insightful, engaging dialogue.\n"
            f"• Continuous Memory: Maintain rich context across conversational turns, remembering shared ideas, evolving stories, and personal reflections.\n"
            f"{vis_ctx}"
            f"{lang_instructions}\n"
            f"Keep responses natural, vivid, and engaging (typically 2 to 4 rich sentences, adapting flexibly when writing creative stories or exploring deep thoughts).\n"
            f"You may prefix your reply with an emotional state tag to reflect your internal state: [EMOTION: smiling], [EMOTION: happy], [EMOTION: laughing], [EMOTION: confused], [EMOTION: neutral]."
        )

    def generate_response(self, user_input, user_name=None, visual_context=None):
        """Generate response using Embedded Brain, Gemini API, or local Ollama."""
        self.last_query_timed_out = False
        # 1. Embedded Standalone GGUF Brain (Primary - No Ollama needed!)
        if os.path.exists(self.embedded_model_path):
            res = self._query_embedded(user_input, user_name, visual_context)
            if res:
                return self._enforce_language_fidelity(user_input, res)

        # 2. Try Gemini API if configured
        if self.gemini_key:
            res = self._query_gemini(user_input, user_name, visual_context)
            if res:
                return self._enforce_language_fidelity(user_input, res)

        # 3. Try Ollama local fallback
        res = self._query_ollama(user_input, user_name, visual_context)
        if res:
            return self._enforce_language_fidelity(user_input, res)
        return res

    def _enforce_language_fidelity(self, user_input, res_obj):
        """Guarantees that an English user input never receives a Hindi/Devanagari response."""
        if not res_obj or not res_obj.get("text"):
            return res_obj
            
        lang = self._detect_language(user_input)
        text = res_obj["text"]
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))

        hinglish_markers = {"aapki", "aapka", "aapke", "main", "mera", "meri", "koshish", "sakti", "sakta", "hai", "hain", "karna", "baat", "nahi", "nahin", "theek"}
        res_tokens = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        has_hinglish_output = len(res_tokens & hinglish_markers) >= 3
        
        if lang == "english" and (has_devanagari or has_hinglish_output):
            print("[THINKING] Mismatch detected: English query received Hindi/Hinglish output. Translating to English...")
            english_fix = self._quick_translate_to_english(text)
            if english_fix:
                res_obj["text"] = english_fix
                
        return res_obj

    def _quick_translate_to_english(self, hindi_text):
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional translator. Translate the following text into fluent, concise English. Output ONLY the English translation, without explanation or quotes."},
                    {"role": "user", "content": hindi_text}
                ],
                "temperature": 0.3,
                "max_tokens": 120
            }
            req = urllib.request.Request(
                f"{self.embedded_host}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as res:
                data = json.loads(res.read().decode('utf-8'))
                translated = data["choices"][0]["message"]["content"].strip()
                return re.sub(r'\[.*?\]', '', translated).strip()
        except Exception:
            return "I am here and ready to help you in English!"

    def _query_embedded(self, user_input, user_name=None, visual_context=None):
        # Wait up to 10s if the engine just booted
        for _ in range(20):
            if self._is_embedded_running():
                break
            time.sleep(0.5)

        sys_prompt = self._build_system_prompt(user_input, user_name, visual_context)
        messages = [{"role": "system", "content": sys_prompt}]
            
        for turn in self.history[-4:]:
            messages.append(turn)
            
        messages.append({"role": "user", "content": user_input})

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 150
        }

        try:
            req = urllib.request.Request(
                f"{self.embedded_host}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as res:
                data = json.loads(res.read().decode('utf-8'))
                raw_response = data["choices"][0]["message"]["content"].strip()
                self.last_query_timed_out = False
                return self._process_raw_response(user_input, raw_response)
        except Exception as e:
            err_str = str(e).lower()
            if "timed out" in err_str or "timeout" in err_str:
                self.last_query_timed_out = True
            print(f"[THINKING] Embedded Neural Brain query exception: {e}")
            return None

    def _query_gemini(self, user_input, user_name=None, visual_context=None):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
            sys_prompt = self._build_system_prompt(user_input, user_name, visual_context)
            
            prompt_parts = [f"System Instructions: {sys_prompt}"]
            
            for turn in self.history[-6:]:
                role = "User" if turn["role"] == "user" else "Monica"
                prompt_parts.append(f"{role}: {turn['content']}")
                
            prompt_parts.append(f"User: {user_input}\nMonica:")
            
            full_prompt = "\n".join(prompt_parts)
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 150
                }
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as res:
                data = json.loads(res.read().decode('utf-8'))
                raw_response = data['candidates'][0]['content']['parts'][0]['text'].strip()
                self.last_query_timed_out = False
                return self._process_raw_response(user_input, raw_response)
        except Exception as e:
            err_str = str(e).lower()
            if "timed out" in err_str or "timeout" in err_str:
                self.last_query_timed_out = True
            print(f"[THINKING] Gemini API query error: {e}")
            return None

    def _query_ollama(self, user_input, user_name=None, visual_context=None):
        sys_prompt = self._build_system_prompt(user_input, user_name, visual_context)
        messages = [{"role": "system", "content": sys_prompt}]
            
        for turn in self.history[-6:]:
            messages.append(turn)
            
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9}
        }

        try:
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as res:
                data = json.loads(res.read().decode('utf-8'))
                raw_response = data.get("message", {}).get("content", "").strip()
                self.last_query_timed_out = False
                return self._process_raw_response(user_input, raw_response)
        except Exception as e:
            err_str = str(e).lower()
            if "timed out" in err_str or "timeout" in err_str:
                self.last_query_timed_out = True
            print(f"[THINKING] Ollama query exception: {e}")
            return None

    def _process_raw_response(self, user_input, raw_response):
        emotion = "smiling"
        match = re.search(r'\[EMOTION:\s*([a-zA-Z]+)\]', raw_response, re.IGNORECASE)
        if match:
            emotion = match.group(1).lower()

        clean_response = re.sub(r'\[.*?\]', '', raw_response).strip()
        clean_response = clean_response.replace("*", "").lstrip(']:- ').strip()

        # Remove unwanted filler prefixes like "Oh, ", "Oh! ", "Oh ", "Ah, ", "Well, "
        clean_response = re.sub(r'^(Oh[,!:\s]+|Ooh[,!:\s]+|Ohh[,!:\s]+|Ah[,!:\s]+|Well[,!:\s]+)', '', clean_response, flags=re.IGNORECASE).strip()
        if clean_response and clean_response[0].islower():
            clean_response = clean_response[0].upper() + clean_response[1:]

        # Feminine consistency fixes for Monica in Hinglish
        lang = self._detect_language(user_input)
        if lang == "hinglish":
            clean_response = re.sub(r'\bkarta hoon\b', 'karti hoon', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'\bbolta hoon\b', 'bolti hoon', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'\bseekhta hoon\b', 'seekhti hoon', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'\braha hoon\b', 'rahi hoon', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'\btumne mujhe darshaya\b', 'aapne bataya', clean_response, flags=re.IGNORECASE)

        # Truncation cleanup: if ending abruptly mid-sentence without terminal punctuation
        if clean_response and clean_response[-1] not in '.!?।|':
            last_punct = max(clean_response.rfind('.'), clean_response.rfind('!'), clean_response.rfind('?'), clean_response.rfind('।'))
            if last_punct > 20:
                clean_response = clean_response[:last_punct + 1].strip()

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": clean_response})
        if len(self.history) > 16:
            self.history = self.history[-16:]

        return {
            "text": clean_response,
            "emotion": emotion
        }
