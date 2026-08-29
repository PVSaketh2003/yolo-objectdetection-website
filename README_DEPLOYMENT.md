# 24/7 365 Permanent Cloud Deployment Guide

To make your website link **work 24/7 worldwide even when your laptop is turned OFF**, the app must be hosted on a **24/7 Cloud Server**.

Below are the 3 fastest, free ways to deploy it in 2 minutes:

---

## 🌟 Option 1: Render.com (Recommended Free 24/7 Hosting)

Render hosts Docker web services 24/7 for free with a permanent `.onrender.com` URL.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "YOLO Studio 24/7 Cloud Deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/yolo-studio.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [Render.com](https://render.com) -> Click **New +** -> **Web Service**.
   - Connect your GitHub repository `yolo-studio`.
   - Environment select: **Docker**.
   - Click **Create Web Service**.

3. **Your 24/7 Link**:
   - Render gives you a permanent HTTPS link:
     👉 `https://yolo-studio.onrender.com`
   - **Works 24/7/365 worldwide even when your Mac is turned off!**

---

## 🚀 Option 2: Hugging Face Spaces (Free 24/7 GPU/CPU Hosting)

Hugging Face provides free 24/7 Docker spaces with instant worldwide access.

### Steps:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) -> Click **Create new Space**.
2. Select SDK: **Docker**.
3. Upload/push your project files.
4. Hugging Face builds your Docker container and gives you a permanent 24/7 link:
   👉 `https://huggingface.co/spaces/YOUR_USERNAME/yolo-studio`

---

## ⚡ Option 3: Railway.app / Fly.io

1. Sign up at [Railway.app](https://railway.app).
2. Click **Deploy from GitHub repo**.
3. Select your `yolo-studio` repository -> Railway automatically detects the `Dockerfile` and deploys it.
4. Gives you a permanent 24/7 link:
   👉 `https://yolo-studio.up.railway.app`

---

## 💡 Local Tunnel vs Cloud Deployment Summary

| Feature | Local Tunnel (`localtunnel`/`cloudflared`) | 24/7 Cloud Deployment (Render/Railway/HuggingFace) |
|---|---|---|
| **Requires Laptop ON?** | ❌ Yes (Laptop must stay ON) | ✅ No (Works 24/7 when laptop is OFF) |
| **Link Uptime** | ⏳ Temporary while laptop is open | ♾️ 100% Permanent 24/7/365 |
| **Setup Needed** | `./start.sh` on Mac | Deployed once to GitHub/Render |
