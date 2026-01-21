"""Entry point for GUI application."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit GUI application."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"])


if __name__ == "__main__":
    main()
