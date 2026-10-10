# IM-Milan: Best Time to Call Predictor

AI-powered 15-minute calling window recommendation service for IndiaMART free-to-paid seller conversion.
Integrates live Redash PostgreSQL DWH records with IM-Milan Model 4 (HistGradientBoostingClassifier quarter-hour planner).

## Architecture
- **Frontend:** Responsive, minimalist UI (`templates/index.html`)
- **Backend:** Flask API (`web_ui.py`)
- **Redash DWH:** `https://redash.intermesh.net` (fetches seller firmographics & hotlead timestamps)
- **ML Planner:** Model 4 quarter-hour sequence planner (`im-milan/btc_service/`)

## Quick Start
```bash
pip install -r requirements.txt
python web_ui.py --port 5000
```

## Free Cloud Deployment
- **Render.com:** Connect this repository as a free Web Service (auto-detected via `render.yaml` or `Dockerfile`).
- **Railway:** Deploy directly via Dockerfile.
- **Cloudflare Tunnel:** `cloudflared tunnel --url http://localhost:5000`
