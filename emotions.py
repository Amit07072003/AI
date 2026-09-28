class Emotions:
    STATES = ["neutral", "happy", "confused", "annoyed", "angry", "laughing", "smiling"]
    
    def __init__(self):
        self.current_state = "neutral"
        self.confusion_count = 0
        self.rejection_count = 0
        self.rule_violation_count = 0

    def trigger_success(self):
        """User thanked the AI or taught it something."""
        self.current_state = "smiling"
        self.confusion_count = 0
        self.rejection_count = 0

    def trigger_confusion(self):
        """AI encountered an unknown intent or concept."""
        self.confusion_count += 1
        if self.confusion_count >= 2:
            self.current_state = "confused"

    def trigger_rejection(self):
        """User denied permission (Rule 1/2)."""
        self.rejection_count += 1
        if self.rejection_count >= 2:
            self.current_state = "annoyed"
            
    def trigger_anger(self):
        """User violated rules explicitly."""
        self.rule_violation_count += 1
        self.current_state = "angry"
        
    def trigger_joke(self):
        """User said something funny."""
        self.current_state = "laughing"

    def trigger_reset(self):
        """Reset to neutral over time or context switch."""
        self.current_state = "neutral"

    def step_decay(self):
        """Naturally decay intense or transient emotions back to neutral/smiling."""
        if self.current_state in ["laughing", "angry", "annoyed", "confused"]:
            self.current_state = "smiling" if self.current_state == "laughing" else "neutral"

    def modify_response(self, base_response):
        """Returns clean response while allowing the UI 3D avatar to use the emotion state."""
        # Decay transient emotion for the next turn
        self.step_decay()
        return base_response

