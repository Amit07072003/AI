import speech_recognition as sr
from biometrics import VoiceBiometrics
import os

def enroll():
    bio = VoiceBiometrics()
    if not bio.is_available():
        print("ERROR: Math libraries missing!")
        print("Please run this command first:")
        print("pip install numpy librosa soundfile")
        return

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("========================================")
        print("   VOICE BIOMETRICS ENROLLMENT SETUP")
        print("========================================")
        print("Adjusting for ambient noise... (Keep quiet for 2 seconds)")
        r.adjust_for_ambient_noise(source, duration=2)
        
        print("\nPlease say the following phrase clearly in your normal voice:")
        print("\n--> 'I am the authorized user of this system.' <--\n")
        print("Listening...")
        
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=5)
        except Exception as e:
            print(f"Error capturing audio: {e}")
            return
            
        temp_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_enroll.wav")
        with open(temp_wav, "wb") as f:
            f.write(audio.get_wav_data())
            
        print("Extracting vocal footprint (MFCCs)...")
        bio.enroll(temp_wav)
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        print("\nSUCCESS! Your voice footprint has been mathematically secured.")
        print("The AI will now ONLY respond to your voice.")

if __name__ == "__main__":
    enroll()
