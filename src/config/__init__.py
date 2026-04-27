"""src.config — re-exports everything from settings so existing
`from src.config import DATASET_DIR` etc. keeps working."""
from .settings import *   # noqa: F401, F403
