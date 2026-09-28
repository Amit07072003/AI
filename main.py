import sys
import os

# Ensure full UTF-8 Unicode encoding for terminal and string output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import builtins
from ui_server import start_ui_server, update_ui_state, has_input, get_input, log_thought, log_chat

# Hijack print to capture logs for the UI Dashboard
_original_print = builtins.print
def custom_print(*args, **kwargs):
    try:
        text = " ".join(str(a) for a in args)
        if "[THINKING]" in text or "[EMOTION" in text or "[SECURITY" in text or "[HEARD]" in text or "AI:" in text or "[VISION" in text:
            log_thought(text)
        else:
            _original_print(*args, **kwargs)
    except Exception:
        pass
builtins.print = custom_print

from brain import Brain
from voice import VoiceInterface
from vision import VisionSystem
from rules import PrimeDirectives
import time
import threading
from ui_server import input_queue

def main():
    _original_print("==================================================")
    _original_print("Humanoid Brain AI - Powered by LLaMA 3.2 Neural Core")
    _original_print("Status: Neural Reasoning ACTIVE (Ollama: llama3.2)")
    _original_print("Status: Prime Directives ACTIVE")
    _original_print("Status: Web Dashboard ACTIVE")
    _original_print("Status: Live Chat & Voice ACTIVE")
    _original_print("==================================================\n")
    
    start_ui_server()
    time.sleep(1) 
    
    voice = VoiceInterface()
    vision = VisionSystem()
    ai_brain = Brain(vision=vision)
    
    if vision.enabled:
        _original_print("[System] Vision system in STANDBY (activates on-demand when CAMERA button is pressed or visual query received).")
    else:
        _original_print("[System] Vision system not available (no OpenCV).")
        
    _original_print("\n[System] >>> Open your browser to http://localhost:8001 to use the Dashboard! <<<\n")
    
    # Start background microphone listener
    def listen_loop():
        from ui_server import app_state, update_ui_state
        while True:
            if app_state.get("mic_muted", False) or voice.is_speaking:
                update_ui_state(is_listening=False)
                time.sleep(0.3)
                continue
                
            update_ui_state(is_listening=True)
            text = voice.listen()
            update_ui_state(is_listening=False)
            
            # If Monica was speaking during listening, discard to avoid self-echo feedback
            if voice.is_speaking:
                continue

            if text:
                input_queue.append({"text": text, "source": "voice"})
            time.sleep(0.1)
            
    if voice.stt_enabled:
        mic_thread = threading.Thread(target=listen_loop, daemon=True)
        mic_thread.start()
        _original_print("[System] Voice recognition is ACTIVE. Speak to Monica!")
    else:
        _original_print("[System] Voice libraries not found. Microphone is disabled.")
    
    # Start background terminal keyboard listener
    def terminal_input_loop():
        while True:
            try:
                line = sys.stdin.readline()
                if line:
                    line = line.strip()
                    if line:
                        input_queue.append({"text": line, "source": "text"})
                else:
                    time.sleep(1)
            except Exception:
                break

    term_thread = threading.Thread(target=terminal_input_loop, daemon=True)
    term_thread.start()
    _original_print("[System] Terminal keyboard input is ACTIVE. (Type commands or chat in text anytime).\n")
    
    update_ui_state(emotion=ai_brain.emotions.current_state, text="Booting complete. Ready for commands.", needs_permission=False)
    
    while True:
        try:
            if has_input():
                user_input, input_source = get_input()
                user_input = user_input.strip()
                if not user_input:
                    continue
                
                if user_input == "CMD_MUTE_MIC":
                    update_ui_state(mic_muted=True)
                    log_thought("[SYSTEM] Microphone muted by user.")
                    continue
                elif user_input == "CMD_UNMUTE_MIC":
                    update_ui_state(mic_muted=False)
                    log_thought("[SYSTEM] Microphone unmuted by user.")
                    continue
                # Obedience Check: Prime Directive 4 (STOP / HALT)
                if PrimeDirectives.is_stop_command(user_input) or user_input.strip().upper() == "STOP":
                    print("[THINKING] Prime Directive 4 (Obedience) triggered. Halting system.")
                    update_ui_state(emotion="neutral", text="SYSTEM HALTED. Goodbye.", speaking=False)
                    log_chat("AI", "SYSTEM HALTED. Goodbye.")
                    if voice.tts_enabled and input_source == "voice":
                        voice.speak("System halted. Goodbye.")
                    vision.stop()
                    time.sleep(0.5)
                    _original_print("[System] Monica successfully terminated.")
                    os._exit(0)

                clean_cmd = user_input.lower().strip()

                # On-demand Camera Hardware Control
                if user_input == "CMD_START_CAMERA" or clean_cmd in ["open camera", "start camera", "turn on camera"]:
                    vision.start()
                    log_thought("[SYSTEM] Camera activated by user.")
                    update_ui_state(text="Camera activated. Visual tracking online.")
                    continue
                elif user_input == "CMD_STOP_CAMERA" or clean_cmd in ["close camera", "stop camera", "turn off camera"]:
                    vision.stop()
                    log_thought("[SYSTEM] Camera deactivated by user.")
                    update_ui_state(text="Camera deactivated.")
                    continue

                # Lock visual gaze and turn face toward the active speaker in the room
                active_spk = None
                if vision.is_running:
                    active_spk = vision.identify_and_look_at_speaker()

                # Check for explicit directional gaze commands
                if clean_cmd in ["look left", "turn left", "look to the left"]:
                    vision.look_direction("left")
                    response = "Looking toward the left user."
                    print(f"AI: {response}")
                    log_chat("AI", response)
                    update_ui_state(emotion="smiling", text=response, speaking=False)
                    if input_source == "voice" and voice.tts_enabled:
                        voice.speak(response)
                    continue
                elif clean_cmd in ["look right", "turn right", "look to the right"]:
                    vision.look_direction("right")
                    response = "Looking toward the right user."
                    print(f"AI: {response}")
                    log_chat("AI", response)
                    update_ui_state(emotion="smiling", text=response, speaking=False)
                    if input_source == "voice" and voice.tts_enabled:
                        voice.speak(response)
                    continue
                elif clean_cmd in ["look center", "look straight", "look forward", "look at me"]:
                    vision.look_direction("center")
                    response = "Looking straight at you."
                    print(f"AI: {response}")
                    log_chat("AI", response)
                    update_ui_state(emotion="smiling", text=response, speaking=False)
                    if input_source == "voice" and voice.tts_enabled:
                        voice.speak(response)
                    continue

                log_chat("You", user_input)
                update_ui_state(text="Thinking...")
                
                response = ai_brain.process_input(user_input)
                is_timeout = getattr(ai_brain, 'last_timed_out', False)
                needs_perm = (ai_brain.state == "awaiting_internet_consent" or ai_brain.state == "awaiting_memory_consent")
                
                print(f"AI: {response}")
                log_chat("AI", response, timed_out=is_timeout)
                
                # Modality matching: Only speak aloud if the user asked via voice
                if input_source == "voice" and voice.tts_enabled:
                    log_thought(f"[THINKING] Voice input detected -> Speaking response in audio.")
                    update_ui_state(emotion=ai_brain.emotions.current_state, text=response, speaking=True, needs_permission=needs_perm)
                    voice.speak(response)
                    while voice.is_speaking:
                        if has_input():
                            cmd, cmd_src = get_input()
                            if cmd == "CMD_MUTE_MIC": update_ui_state(mic_muted=True)
                            elif cmd == "CMD_UNMUTE_MIC": update_ui_state(mic_muted=False)
                            elif cmd == "STOP" or cmd == "CMD_STOP": break
                        time.sleep(0.1)
                    update_ui_state(speaking=False)
                else:
                    # Text response: purely visual/textual, no audio output
                    log_thought(f"[THINKING] Text input detected -> Outputting text-only response (Audio muted).")
                    update_ui_state(emotion=ai_brain.emotions.current_state, text=response, speaking=False, needs_permission=needs_perm)
                
                if "SYSTEM HALTED" in response and "SHUTDOWN_SERVER" in user_input:
                    break
            else:
                time.sleep(0.1) # Prevent CPU hogging
                
        except KeyboardInterrupt:
            print("[THINKING] Interrupt received. Halting.")
            update_ui_state(emotion="neutral", text="System halted.")
            break
        except Exception as e:
            print(f"[ERROR] An unexpected system error occurred: {e}")
            update_ui_state(emotion="confused", text="An error occurred.")

if __name__ == "__main__":
    main()
