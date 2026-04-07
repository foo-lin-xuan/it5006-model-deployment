# Fraud Detection App - Deploy to Streamlit Cloud

This folder contains everything needed to deploy the Streamlit frontend to Streamlit Cloud.

## Quick Deploy (5 Steps)

### Step 1: Create GitHub Repository

```bash
# Create new repo on GitHub named 'fraud-detection-app'
# Then push this folder:

cd Deploy_Streamlit
git init
git add .
git commit -m "Initial commit - Streamlit fraud detection app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-app.git
git push -u origin main
```

### Step 2: Go to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"** or **"Sign in"**
3. Select **"Sign in with GitHub"**

### Step 3: Deploy New App

1. Click **"New app"**
2. Select your repository: `fraud-detection-app`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy!"**

### Step 4: Configure Environment Variable (Optional)

If using a different API URL:

1. Click **"Advanced settings"** before deploying
2. Add secret:
   ```
   API_URL = "https://your-api-url.onrender.com"
   ```

By default, it uses: `https://model-deployment2026.onrender.com`

### Step 5: Wait for Deployment

Streamlit Cloud will:
1. Clone your repository
2. Install dependencies from `requirements.txt`
3. Start your app

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## Files in This Folder

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Local Testing

Before deploying, test locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Important Notes

### Free Tier Behavior

- **Render.com API**: May sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- The app handles this with timeout warnings

### Customization

To use your own API:
1. Deploy FastAPI to Render (see `Deploy_Render` folder)
2. Update `API_URL` in `app.py` or set as environment variable

---

## Troubleshooting

### "Cannot connect to API"
- The Render API may be sleeping
- Wait 30-60 seconds and try again

### "Timeout" errors
- Free tier APIs wake slowly
- The app has 30-second timeouts built in

### App not updating
- Push changes to GitHub
- Streamlit Cloud auto-deploys on push
