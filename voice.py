import asyncio
import os
import ctypes
import threading
import time

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from biometrics import VoiceBiometrics
except ImportError:
    class VoiceBiometrics:
        def __init__(self, *args, **kwargs):
            pass
        def is_available(self):
            return False

class VoiceInterface:
    def __init__(self):
        self.tts_enabled = False
        self.stt_enabled = False
        self.biometrics = VoiceBiometrics()
        self.recognizer = None
        
        self.is_speaking = False
        self.interrupted = False
        
        if HAS_EDGE_TTS or HAS_WIN32COM or pyttsx3 is not None:
            self.tts_enabled = True
            
        if sr is not None:
            try:
                self.recognizer = sr.Recognizer()
                self.stt_enabled = True
            except Exception:
                pass

    def _speak_thread(self, text):
        self.is_speaking = True
        self.interrupted = False
        
        try:
            import re
            # Clean bracketed stage notes (e.g. [Amit, ...], [EMOTION: ...]) and emojis
            clean_text = re.sub(r'\[.*?\]', '', text)
            clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)
            clean_text = re.sub(r'[\u2600-\u27bf]', '', clean_text)
            clean_text = clean_text.strip()
            if not clean_text:
                return

            played = False
            # 1. Try Microsoft Neural Voice via Edge-TTS (Natural Hindi & English)
            if HAS_EDGE_TTS:
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    mp3_file = os.path.join(base_dir, f"speech_{int(time.time() * 1000) % 10000}.mp3")
                    
                    async def _generate():
                        communicate = edge_tts.Communicate(clean_text, voice_name)
                        await communicate.save(mp3_file)
                        
                    asyncio.run(_generate())
                    
                    if os.path.exists(mp3_file) and not self.interrupted:
                        playsound.playsound(mp3_file, block=True)
                        try:
                            os.remove(mp3_file)
                        except Exception:
                            pass
                        played = True
                except Exception as e:
                    print(f"[VOICE] Edge-TTS error: {e}")
                    played = False

            # 2. Fallback to Windows SAPI
            if not played:
                import pythoncom
                pythoncom.CoInitialize()
                
                if HAS_WIN32COM:
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    try:
                        voices = speaker.GetVoices()
                        for i in range(voices.Count):
                            desc = voices.Item(i).GetDescription().lower()
                            if "zira" in desc or "female" in desc or "eva" in desc or "hindi" in desc:
                                speaker.Voice = voices.Item(i)
                                break
                    except Exception:
                        pass
                        
                    speaker.Speak(clean_text)
                elif pyttsx3 is not None:
                    engine = pyttsx3.init()
                    engine.say(clean_text)
                    engine.runAndWait()
        except Exception as e:
            print(f"[VOICE] Synthesis error: {e}")
        finally:
            time.sleep(0.3)  # Small cooldown so microphone does not pick up trailing room echo
            self.is_speaking = False

    def speak(self, text):
        if self.tts_enabled:
            self.is_speaking = True
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def listen(self):
        if not self.stt_enabled or sr is None:
            return ""
            
        try:
            with sr.Microphone() as source:
                print("\n[HEARD] Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                try:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)
                    
                    # Check Voice Biometrics
                    if self.biometrics.enrolled_mfcc is not None:
                        temp_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_auth.wav")
                        with open(temp_wav, "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        is_auth = self.biometrics.verify(temp_wav)
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                            
                        if not is_auth:
                            print("\n[SECURITY] UNAUTHORIZED VOICE DETECTED. IGNORING COMMAND.")
                            return ""
                    
                    try:
                        text = self.recognizer.recognize_google(audio)
                    except Exception:
                        try:
                            text = self.recognizer.recognize_sphinx(audio)
                        except Exception:
                            text = ""

                    if text:
                        print(f"[HEARD] You said: {text}")
                    
                    # BARGE-IN DETECTION
                    if text.strip() and self.is_speaking:
                        self.interrupted = True
                        print("[SYSTEM] BARGE-IN DETECTED! Stopping speech.")
                        
                    return text
                except sr.WaitTimeoutError:
                    return ""
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError as e:
                    return ""
        except Exception as e:
            # Microphone hardware not initialized or PyAudio not present
            return ""
