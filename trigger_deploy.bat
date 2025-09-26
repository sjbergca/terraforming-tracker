@echo off
echo Triggering Render deployment...
curl -s -X POST "https://api.render.com/deploy/srv-d1g2q7vfte5s7385vmag?key=HPO-mAgmdKg" >nul
if %errorlevel%==0 (
    echo ✅ Deploy triggered successfully.
) else (
    echo ❌ Failed to trigger deploy.
)
