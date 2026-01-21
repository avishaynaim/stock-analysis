"""Stock Analysis GUI application."""

__all__ = ["run_app"]


def run_app():
    """Run the Streamlit GUI application."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
