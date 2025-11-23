import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ถ้าไม่มีค่าใน env จะใช้ default (ควรตั้งค่าใน Production environment)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")