import os


def clean(verbose=False):
    """
    Clean all saved pickle files. Useful after updating dependencies.
    """
    root = os.path.dirname(__file__)
    num = 0
    for root, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".pkl"):
                os.remove(os.path.join(root, file))
                if verbose:
                    print("Removing %s" % os.path.join(root, file))
                num += 1
    if verbose:
        if num == 1:
            print("Removed %i file" % num)
        else:
            print("Removed %i files" % num)
