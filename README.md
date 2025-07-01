# Terraforming Mars Game Tracker

This project is a web-based dashboard that visualizes and tracks two-player Terraforming Mars game data. It is built using **Python**, **Dash**, and **Plotly**, and deployed via **Render** for easy public access.

---

## 🌐 Live Web Dashboard

You can view the dashboard live at:  https://terraforming-tracker.onrender.com/

👉 **[Render Dashboard](https://dashboard.render.com/web/)**

> ⚠️ Note: You must **log in to your Render account** to manage the deployment. The live site URL is accessible via your Render service.

---

## 🚀 How to Run the App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/terraforming-tracker.git
cd terraforming-tracker
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

---

## 🔄 How to Upload Updated Game Data

1. Launch the web app.
2. Use the **"Upload Spreadsheet"** drag-and-drop area at the top.
3. Upload your `.xlsx` game tracking file. It will update the dashboard in real time.
4. The file persists until you upload another.

---

## 🧠 GitHub Instructions (Pushing Your Code Changes)

To save and push your code (including Dash app or VBA macros, etc.):

```bash
git status                    # check changes
git add .                     # stage changes
git commit -m "Your message"  # commit changes
git push                      # push to GitHub
```

You can do this manually or use an IDE like **VS Code** or **GitHub Desktop**.

---

## 📋 Updating Game Data via Copy Script (Preferred)

Use this method if you're copying from a **master Excel file** (e.g., in OneDrive) to the working app folder (`games/games.xlsx`), and want to commit & push it automatically.

### ✅ Steps:

1. Ensure **Excel is closed** — the file cannot be copied if it's open!
2. Double-click the script:
   ```
   update_games_and_push.bat
   ```
   This will:
   - Copy the latest spreadsheet from your OneDrive
   - Overwrite `games/games.xlsx`
   - Commit and push the update to GitHub

### ⚠️ Important:

- This script **must be run from your office desktop**.
- Your Git credentials must be cached or authenticated on that machine.

---

## 📁 Project Structure

```
terraforming_tracker/
├── app.py                      # Main Dash app
├── copy_data.py                # Copies updated Excel file and commits
├── update_games_and_push.bat   # Single-click script for updates
├── requirements.txt            # Python dependencies
├── games/
│   └── games.xlsx              # Working copy of the spreadsheet
└── README.md                   # You are here!
```

---

## 📝 Future Ideas

- [ ] Highlight map options stats
- [ ] Player filters on graphs and tables
- [ ] Export filtered data
- [ ] Add tag system for notes (e.g., draft used, promos, etc.)

---



