Terraforming Mars Tracker
A custom-built dashboard for tracking head-to-head games of Terraforming Mars between SB and AV. Built with Dash and Plotly.

🌐 Launch the App on Render
Go to Render Dashboard

Select your service (e.g. terraforming-tracker)

Click "Open in Browser" to launch the live dashboard

✅ The app auto-reloads when you push updates to the GitHub repo.

🔁 Updating Data (CSV Method)
Save your latest spreadsheet as:

bash
Copy
Edit
games/games.csv
Use the included batch script (e.g. copy_data.bat) to overwrite the existing file:

Make sure Excel is closed

Run the script from your office desktop (not your laptop)

Push the updated data to GitHub (see GitHub instructions below)

🔧 Local Testing Instructions
To test locally without Render:

(One-time) Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Then open your browser to http://127.0.0.1:8050

🛠 GitHub Instructions
To commit and push local changes to GitHub:

bash
Copy
Edit
git add .
git commit -m "💾 Updated data and dashboard"
git push
To sync your local copy with any online GitHub changes:

bash
Copy
Edit
git pull
If you get a merge error, try:
git stash, git pull, then git stash pop

⚙️ Performance Tips
Using .csv instead of .xlsx drastically reduces memory usage

Avoid loading unnecessary tabs or hidden sheets in Excel

Keep only the required data columns and games in games.csv

📊 Dashboard Features
Cumulative win % over time

Score distributions (PDF, CDF, combined scores, margins)

Player summaries including win streaks and score ranges

Corp vs Corp matchup matrix

Corporation performance by map

Game results table with sorting/filtering

