import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Prefer loading .env from the repo root; fall back to backend/.env then default
ROOT_DIR = Path(__file__).resolve().parents[2]
env_root = ROOT_DIR / ".env"
env_backend = Path(__file__).resolve().parents[1] / ".env"
if env_root.exists():
	load_dotenv(env_root)
elif env_backend.exists():
	load_dotenv(env_backend)
else:
	load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if SUPABASE_URL and SUPABASE_SECRET_KEY:
	supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
else:
	class _MissingSupabase:
		def table(self, *args, **kwargs):
			raise RuntimeError(
				"Supabase client not configured. Copy .env.example to .env and set SUPABASE_URL and SUPABASE_SECRET_KEY in the repo root."
			)

	supabase = _MissingSupabase()