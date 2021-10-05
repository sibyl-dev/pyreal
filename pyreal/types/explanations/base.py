class ExplanationType:
    def __init__(self, explanation):
        self.validate_wrapper(explanation)
        self.explanation = explanation

    def get(self):
        return self.explanation

    @staticmethod
    def validate(explanation):
        pass

    @classmethod
    def validate_wrapper(cls, explanation):
        cls.validate(explanation)
