from pyreal.explanation_types import Explanation


class NarrativeExplanation(Explanation):
    def validate(self):
        if not isinstance(self.explanation, list):
            raise AssertionError("Explanation must be a list of strings")
        for instance in self.explanation:
            assert isinstance(instance, str), "Explanation must be a list of strings"
