"""Root entrypoint for Streamlit Community Cloud deployments.
"""

import os
import sys
from pathlib import Path

# Set root directory in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main dashboard
dashboard_file = ROOT_DIR / "app" / "dashboard.py"
with open(dashboard_file, encoding="utf-8") as f:
    code = compile(f.read(), str(dashboard_file), "exec")
    exec(code, globals())

