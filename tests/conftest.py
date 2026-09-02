from pathlib import Path

from dotenv import load_dotenv

# Loaded once for the whole test session so `pytest` (unit or integration)
# never needs a manual `set -a; source .env; set +a` first — matches how
# compose.yml's `env_file: .env` already works for the container.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
