import os


def clean():
    """
    Clean all saved pickle files. Useful after updating dependencies.
    """
    root = os.path.dirname(__file__)
    for root, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".pkl"):
                os.remove(os.path.join(root, file))
