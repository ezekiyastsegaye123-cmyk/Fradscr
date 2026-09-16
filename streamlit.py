"""
FRADSCR Streamlit Application Entrypoint
========================================
Production entrypoint for the FRADSCR Drought Early Warning and Solar
Groundwater Advisory System with full Model-2 (model-2.ipynb) acceptance.

Usage:
    streamlit run streamlit.py
    streamlit run streamlit_app.py
    python streamlit.py
"""
import sys
from pathlib import Path
import importlib.util

_current_dir = str(Path(__file__).resolve().parent)

# When imported as `import streamlit`, dynamically forward to the installed library
if __name__ == "streamlit":
    _temp_path = [p for p in sys.path if p != _current_dir and p != ""]
    _real_streamlit = None
    for _p in _temp_path:
        _candidate = Path(_p) / "streamlit" / "__init__.py"
        if _candidate.exists():
            _spec = importlib.util.spec_from_file_location(
                "streamlit",
                _candidate,
                submodule_search_locations=[str(_candidate.parent)],
            )
            if _spec and _spec.loader:
                _real_streamlit = importlib.util.module_from_spec(_spec)
                sys.modules["streamlit"] = _real_streamlit
                _spec.loader.exec_module(_real_streamlit)
                break
    if _real_streamlit is not None:
        globals().update(_real_streamlit.__dict__)

else:
    # When executed as a script (e.g. `streamlit run streamlit.py` or `python streamlit.py`)
    PROJECT_ROOT = Path(__file__).resolve().parent
    if "streamlit.runtime" in sys.modules or "streamlit" in sys.modules.get("__main__", "").__file__:
        # Running inside streamlit execution engine
        app_file = PROJECT_ROOT / "streamlit_app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            code = compile(f.read(), str(app_file), "exec")
            exec(code, globals())
    else:
        # Executed as `python streamlit.py` (optional args: 1 for Model-1, 2 for Model-2)
        import subprocess
        if len(sys.argv) > 1 and sys.argv[1] == "1":
            app_path = PROJECT_ROOT / "app_model_1.py"
            extra_args = sys.argv[2:]
        elif len(sys.argv) > 1 and sys.argv[1] == "2":
            app_path = PROJECT_ROOT / "app_model_2.py"
            extra_args = sys.argv[2:]
        else:
            app_path = PROJECT_ROOT / "streamlit_app.py"
            extra_args = sys.argv[1:]
        subprocess.run(["streamlit", "run", str(app_path)] + extra_args)
