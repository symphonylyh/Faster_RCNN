# rpn package will need ../config.py
import os, sys
from pathlib import Path
path = Path(os.path.abspath(__file__))
sys.path.append(str(path.parent.parent))
