import os
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None

class VoiceBiometrics:
    def __init__(self, print_file=None):
        if print_file is None:
            print_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_print.npy")
        self.print_file = print_file
        self.enrolled_mfcc = None
        self.load_print()

    def is_available(self):
        return librosa is not None

    def load_print(self):
        if os.path.exists(self.print_file):
            self.enrolled_mfcc = np.load(self.print_file)
            print("[BIOMETRICS] Loaded authorized voice print.")

    def extract_features(self, wav_file):
        if not self.is_available():
            raise Exception("librosa not installed. Run: pip install librosa soundfile")
        
        # Load audio file (resample to 16kHz for consistency)
        y, sr = librosa.load(wav_file, sr=16000)
        
        # Extract 13 Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Average the features over time to get a single 13-Dimensional vector
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

    def enroll(self, wav_file):
        features = self.extract_features(wav_file)
        np.save(self.print_file, features)
        self.enrolled_mfcc = features
        print("[BIOMETRICS] Voice print saved successfully.")

    def verify(self, wav_file, threshold=0.85):
        if self.enrolled_mfcc is None:
            print("[BIOMETRICS] No voice print enrolled. Skipping verification.")
            return True

        features = self.extract_features(wav_file)
        
        # Calculate Cosine Similarity between the live voice and saved voice print
        dot_product = np.dot(features, self.enrolled_mfcc)
        norm_a = np.linalg.norm(features)
        norm_b = np.linalg.norm(self.enrolled_mfcc)
        
        if norm_a == 0 or norm_b == 0:
            return False
            
        similarity = dot_product / (norm_a * norm_b)
        
        print(f"[BIOMETRICS] Voice Match Score: {similarity:.2f} (Threshold: {threshold})")
        return similarity >= threshold
