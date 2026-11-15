
from alerts import test_telegram

if test_telegram():
    print("Telegram test successful!")
else:
    print("Telegram test failed. Check .env and telegram.log")
