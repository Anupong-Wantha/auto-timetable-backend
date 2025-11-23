from supabase import create_client
from config import Config

# สร้างตัวแปร supabase ให้ไฟล์อื่นเรียกใช้
try:
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    print("✅ Supabase connected successfully")
except Exception as e:
    print(f"❌ Supabase connection failed: {e}")
    supabase = None