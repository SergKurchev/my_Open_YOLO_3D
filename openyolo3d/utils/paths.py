import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Finds the project root by looking for pyproject.toml or .git.
    """
    current_path = Path(__file__).resolve().parent
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback to the current directory if not found
    return Path(os.getcwd())

def resolve_path(path: str) -> str:
    """
    Resolves a path relative to the project root.
    If the path is already absolute, it returns it as is.
    """
    if os.path.isabs(path):
        return path
    
    root = get_project_root()
    return str(root / path)
