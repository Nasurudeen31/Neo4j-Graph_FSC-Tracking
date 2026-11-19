"""Compatibility entry point for the expanded food-traceability prototype.

The operational implementation now lives in app.py so the command-line,
evaluation, and journal experiment use the same retrieval and validation code.
"""

from app import main


if __name__ == "__main__":
    main()
