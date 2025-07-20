import sys
from pathlib import Path

# Add source directories to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
config_path = project_root / "config"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))
