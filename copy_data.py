import pandas as pd
import time
import os
import subprocess

source = r"C:/Users/sjber/OneDrive/Personal/Games/TMF/01_MulitPlayer_Tracking_Dec2022.xlsx"
csv_destination = r"D:/Research/terraforming_tracker/games/games.csv"
sheet_name = "restart_Mar_2025"

# Wait if file is open and locked
attempts = 5
for attempt in range(attempts):
    try:
        df = pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
        df.to_csv(csv_destination, index=False)
        print("✅ Excel sheet converted and saved as CSV.")
        break
    except PermissionError:
        print(f"⏳ Attempt {attempt + 1}: File is in use. Retrying...")
        time.sleep(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        break

# Git operations
try:
    os.chdir("D:/Research/terraforming_tracker")
    subprocess.run(["git", "add", "games/games.csv"], check=True)
    subprocess.run(["git", "commit", "-m", "🔄 Auto-update games.csv"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("🚀 Changes committed and pushed to GitHub.")
except subprocess.CalledProcessError as e:
    print(f"❌ Git command failed: {e}")
