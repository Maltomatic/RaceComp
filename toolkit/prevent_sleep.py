from datetime import datetime
import time

# run in background from second terminal while running training

while(1):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    time.sleep(300)