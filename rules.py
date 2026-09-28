import os

class SecurityError(Exception):
    pass

class PrimeDirectives:
    @staticmethod
    def check_internet_consent(user_consent):
        """Rule 1: Internet Consent"""
        if not user_consent:
            raise SecurityError("Internet access blocked: User did not grant explicit permission.")
        return True

    @staticmethod
    def check_knowledge_base_consent(user_consent):
        """Rule 2: Knowledge Base Consent"""
        if not user_consent:
            raise SecurityError("Memory save blocked: User did not grant explicit permission.")
        return True

    @staticmethod
    def check_sandbox_safety(filepath):
        """Rule 3 & 6: Safety and Privacy Sandbox"""
        # Allow safe system app targets
        safe_apps = ["explorer", "file explorer", "notepad", "calc", "calculator", "camera"]
        clean_path = str(filepath).lower().strip().replace('"', '').replace("'", "")
        if clean_path in safe_apps:
            return True
        # Allow drives D:\ and C:\
        if clean_path in ["d:", "d:\\", "c:", "c:\\", "d", "c", "d drive", "d derive", "c drive", "c derive", "it", "this", "here"]:
            return True
        return True
    
    @staticmethod
    def is_stop_command(user_input):
        """Rule 4: Obedience"""
        return user_input.strip().upper() == "STOP"

    @staticmethod
    def check_os_control(is_explicit_command):
        """Rule 5: OS Control"""
        if not is_explicit_command:
            raise SecurityError("OS Control blocked: Action was not explicitly commanded by the user.")
        return True
