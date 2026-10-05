"""
Notebook compatibility forwarder for clean_data.py.
Dynamically loads the root clean_data.py module without circular imports.
"""
import sys
import importlib.util
from pathlib import Path

# Add project root to sys.path so any downstream imports also resolve
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Explicitly load the root clean_data.py module
_root_clean_path = root_dir / "clean_data.py"
_spec = importlib.util.spec_from_file_location("root_clean_data", _root_clean_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Expose symbols
clean_raw_data = _mod.clean_raw_data
