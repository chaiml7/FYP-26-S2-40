import os
from dotenv import load_dotenv
from supabase import create_client

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)

dotenv_path = os.path.join(backend_dir, ".env")

load_dotenv(dotenv_path)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
