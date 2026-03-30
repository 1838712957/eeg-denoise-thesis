import os


def pick_existing_path(*paths):
    """Return first existing path; fallback to first candidate."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

