class ViolationEngine:

    def __init__(self):
        pass

    def evaluate(self, action_result):

        if action_result is None:
            return "ANALYZING"

        action = action_result["action"]
        confidence = action_result["confidence"]

        if confidence < 0.6:
            return "NORMAL"

        action = action.lower()

        if "sleep" in action or "lying" in action:
            return "SLEEPING"

        if "phone" in action or "text" in action:
            return "PHONE_USE"

        if "sitting" in action:
            return "IDLE"

        return "NORMAL"
