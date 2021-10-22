class ExplanationType:
    def __init__(self, explanation):
        self.explanation = explanation
        self.validate()

    def get(self):
        return self.explanation

    def validate(self):
        pass
