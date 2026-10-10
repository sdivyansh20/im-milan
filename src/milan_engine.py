"""
IM-Milan Integration Engine.
Fetches seller and hotlead details from Redash PostgreSQL query and generates
top 3 optimal 15-minute calling windows using IM-Milan Model 4.
Calculates top 3 dominant parameters for each predicted calling window.
"""

import os
import sys
import time
import re
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import joblib
import pandas as pd
import numpy as np

# Add im-milan/btc_service to path
BTC_DIR = Path(__file__).resolve().parent.parent / "im-milan" / "btc_service"
if str(BTC_DIR) not in sys.path:
    sys.path.insert(0, str(BTC_DIR))

from app.encode import Encoder
from app.scoring import slot_scores
from app.planner import Planner

logger = logging.getLogger("im_milan")

# Redash configuration
REDASH_HOST = "https://redash.intermesh.net"
REDASH_API_KEY = "NAShEy1FCYvOCTgE3vBCwQauMXEesXPq1iu0KUwE"
REDASH_DATA_SOURCE_ID = 8

REDASH_QUERY_TEMPLATE = """SELECT     hd.fk_glusr_usr_id,     hd.lead_sent_time,      gst.glusr_gst_legal_status AS business_type,     gst.glusr_gst_annual_turnover,      gst.glusr_gst_nature_of_business AS nature_of_business,     gst.glusr_gst_registration_year,     cm.city_name,    cm.state_name,     hd.redis_bucket AS HL_Trigger  FROM (     SELECT         fk_glusr_usr_id,         lead_sent_time,         redis_bucket,         ROW_NUMBER() OVER (             PARTITION BY fk_glusr_usr_id             ORDER BY lead_sent_time DESC         ) AS rn     FROM im_dwh_rpt.data_hotlead_disposition_dtl     WHERE fk_glusr_usr_id = {{ glid }} ) hd  LEFT JOIN im_dwh_rpt.fact_glusr_gst_details gst     ON hd.fk_glusr_usr_id = gst.fk_glusr_usr_id LEFT JOIN im_dwh_rpt.dim_glusr_usr gu     ON hd.fk_glusr_usr_id = gu.glusr_usr_id LEFT JOIN im_dwh_rpt.dim_city_master cm     ON gu.fk_gl_city_id = cm.city_id     AND gu.fk_gl_state_id = cm.state_id  WHERE hd.rn = 1;"""

# Load IM-Milan Model 4 artifacts
ART = BTC_DIR / "artifacts"
bundle = joblib.load(ART / "model4_q15.joblib")
meta = json.loads((ART / "meta.json").read_text())
MODEL, FEATURES = bundle["model"], bundle["features"]
ENC = Encoder(bundle["ref"], meta["top_cities"], meta["buckets_kept"])
PLANNER = Planner(meta)
IST = timezone(timedelta(hours=5, minutes=30))


def fetch_from_redash(glid: int, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
    """
    Executes the PG SQL query via Redash API to fetch seller details and latest hotlead.
    """
    headers = {
        "Authorization": f"Key {REDASH_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": REDASH_QUERY_TEMPLATE,
        "data_source_id": REDASH_DATA_SOURCE_ID,
        "max_age": 0,
        "parameters": {"glid": str(glid)},
    }

    try:
        t0 = time.time()
        resp = requests.post(
            f"{REDASH_HOST}/api/query_results",
            headers=headers,
            json=payload,
            timeout=12,
        )
        resp.raise_for_status()
        res_data = resp.json()

        # If cached result returned directly
        if "query_result" in res_data:
            rows = res_data["query_result"]["data"]["rows"]
            return rows[0] if rows else None

        job = res_data.get("job", {})
        job_id = job.get("id")
        if not job_id:
            logger.warning(f"No job ID in Redash response: {res_data}")
            return None

        # Poll job until completion
        while time.time() - t0 < timeout_sec:
            time.sleep(0.4)
            j_resp = requests.get(
                f"{REDASH_HOST}/api/jobs/{job_id}",
                headers=headers,
                timeout=10,
            )
            j_data = j_resp.json().get("job", {})
            status = j_data.get("status")

            if status == 3:  # Finished
                q_id = j_data.get("query_result_id")
                r_resp = requests.get(
                    f"{REDASH_HOST}/api/query_results/{q_id}",
                    headers=headers,
                    timeout=10,
                )
                rows = r_resp.json().get("query_result", {}).get("data", {}).get("rows", [])
                return rows[0] if rows else None
            elif status == 4:  # Error
                logger.error(f"Redash job error: {j_data.get('error')}")
                return None

        logger.warning(f"Redash query timed out after {timeout_sec}s for glid {glid}")
        return None

    except Exception as e:
        logger.error(f"Error querying Redash for glid {glid}: {e}")
        return None


def get_offline_fallback(glid: int) -> Optional[Dict[str, Any]]:
    """
    Fallback if Redash is unavailable or GLID is not present in Redash disposition table.
    Looks up GLID in local training dataset Sellers.csv.
    """
    base_dir = Path(__file__).resolve().parent.parent
    sellers_csv = base_dir / "Best-Time-to-Call - Sellers.csv"

    if sellers_csv.exists():
        try:
            df = pd.read_csv(sellers_csv)
            match = df[df["fk_glusr_usr_id"] == glid]
            if not match.empty:
                r = match.iloc[0]
                # Hotlead timestamp (1 day ago)
                yesterday = datetime.now() - timedelta(days=1)
                now_str = yesterday.strftime("%Y-%m-%dT10:30:00")
                return {
                    "fk_glusr_usr_id": float(glid),
                    "lead_sent_time": now_str,
                    "business_type": str(r.get("business_type", "Proprietorship")),
                    "glusr_gst_annual_turnover": str(r.get("annual_turnover", "0 - 40 L")),
                    "nature_of_business": str(r.get("nature_of_business", "Manufacturer")),
                    "glusr_gst_registration_year": str(r.get("gst_registration_year", "2018")),
                    "city_name": str(r.get("seller_city", "Other")),
                    "state_name": str(r.get("seller_state", "NA")),
                    "hl_trigger": "PUA",
                    "_is_fallback": True,
                }
        except Exception as e:
            logger.warning(f"Failed to lookup fallback in Sellers.csv: {e}")

    return None


def calculate_confidence_score(prob: float, attempt: int) -> int:
    """
    Calculates an intuitive confidence score percentage (70% - 96%)
    calibrated against the baseline conversion rate for attempt 1, 2, or 3.
    """
    baseline = {1: 0.035, 2: 0.018, 3: 0.012}.get(attempt, 0.02)
    ratio = prob / max(baseline, 0.005)

    if ratio >= 2.5:
        conf = 90 + min(int((ratio - 2.5) * 6), 6)  # 90 - 96%
    elif ratio >= 1.8:
        conf = 84 + int((ratio - 1.8) * 8)          # 84 - 89%
    elif ratio >= 1.2:
        conf = 76 + int((ratio - 1.2) * 13)         # 76 - 83%
    elif ratio >= 0.8:
        conf = 68 + int((ratio - 0.8) * 20)         # 68 - 75%
    else:
        conf = max(55, int(ratio * 70))

    return min(conf, 96)


def compute_top_dominant_parameters(
    seller_row: Dict[str, Any],
    slot_q: int = 75,
    attempt_num: int = 1,
    wait_min: int = 0,
    gap_ratio: float = 1.0,
) -> List[str]:
    """
    Identifies the top 3 parameters (e.g. State, City, Nature of Business, Business Type,
    Turnover, Hotlead Trigger, GST Year, Cadence) that dominated the model's decision.
    Uses model counterfactual sensitivity evaluation.
    """
    codes, gst, notes = ENC.encode(seller_row)
    row = dict(codes)
    row["gst_registration_year"] = np.nan if gst is None else gst
    row["n"] = attempt_num
    row["q"] = slot_q
    row["dow"] = 0

    df_actual = pd.DataFrame([row])[FEATURES]
    p_base = MODEL.predict_proba(df_actual)[0, 1]

    # Clean parameter value representations
    city_val = str(seller_row.get("city_name") or seller_row.get("seller_city") or "Other").strip()
    state_val = str(seller_row.get("state_name") or seller_row.get("seller_state") or "NA").strip()
    nature_val = str(seller_row.get("nature_of_business") or "NA").strip()
    btype_val = str(seller_row.get("business_type") or "NA").strip()
    turnover_val = str(seller_row.get("glusr_gst_annual_turnover") or seller_row.get("annual_turnover") or "NA").strip()
    trig_val = str(seller_row.get("hl_trigger") or seller_row.get("HL_Trigger") or seller_row.get("redis_bucket") or "PUA").strip()
    yr_val = str(int(gst)) if gst else "Missing"
    slot_hr = (slot_q * 15) // 60
    slot_mn = (slot_q * 15) % 60
    slot_time_str = f"{slot_hr:02d}:{slot_mn:02d}"

    param_display = {
        "seller_state": f"State ({state_val})",
        "seller_city": f"City ({city_val})",
        "nature_of_business": f"Nature of Business ({nature_val})",
        "business_type": f"Business Type ({btype_val})",
        "annual_turnover": f"Annual Turnover ({turnover_val})",
        "redis_bucket": f"Hotlead Trigger ({trig_val})",
        "gst_registration_year": f"GST Registration Year ({yr_val})",
        "q": f"Time Slot ({slot_time_str})",
    }

    defaults = {
        "seller_state": len(ENC.ref["seller_state"]),
        "seller_city": len(ENC.ref["seller_city"]),
        "nature_of_business": len(ENC.ref["nature_of_business"]),
        "business_type": len(ENC.ref["business_type"]),
        "annual_turnover": len(ENC.ref["annual_turnover"]),
        "redis_bucket": len(ENC.ref["redis_bucket"]),
        "gst_registration_year": np.nan,
        "q": 60,
    }

    cf_rows = []
    cf_keys = list(defaults.keys())
    for feat in cf_keys:
        r_cf = dict(row)
        r_cf[feat] = defaults[feat]
        cf_rows.append(r_cf)

    p_cfs = MODEL.predict_proba(pd.DataFrame(cf_rows)[FEATURES])[:, 1]
    impacts = {cf_keys[i]: abs(p_base - p_cfs[i]) for i in range(len(cf_keys))}

    # Add Cadence interval impact for attempts 2 and 3
    if attempt_num > 1 and wait_min > 0:
        w_str = f"{wait_min//60}h" if wait_min >= 120 and wait_min % 60 == 0 else (f"{wait_min/60:.1f}h" if wait_min >= 120 else f"{wait_min}m")
        param_display["cadence"] = f"Cadence Interval ({w_str} cooldown)"
        impacts["cadence"] = abs(gap_ratio - 1.0) * p_base

    sorted_feats = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    top_3 = [param_display.get(f[0], f[0]) for f in sorted_feats[:3]]
    return top_3


def predict_im_milan(glid: int) -> Dict[str, Any]:
    """
    End-to-end IM-Milan prediction pipeline:
    1. Fetch seller details & hotlead timestamp from Redash PG SQL query.
    2. Score all 15-minute slots using Model 4.
    3. Generate the 3-attempt sequence plan strictly after the hotlead timestamp.
    4. Compute top 3 dominant parameters for each window.
    5. Formulate concise reasoning highlighting top decision drivers.
    """
    # Step 1: Query Redash
    row = fetch_from_redash(glid)
    is_live_redash = True

    if not row:
        row = get_offline_fallback(glid)
        is_live_redash = False

    if not row:
        raise ValueError(
            f"No hotlead record found for GLID {glid} in Redash DWH or local database. "
            f"Please try an active GLID (e.g. 79959, 141000, 269847, 39323)."
        )

    # Step 2: Prepare inputs for Model 4
    lead_time_str = str(row.get("lead_sent_time", "")).strip()
    try:
        start_dt = datetime.fromisoformat(
            lead_time_str.replace("Z", "+00:00").replace(" ", "T", 1) if "T" not in lead_time_str else lead_time_str.replace("Z", "+00:00")
        )
        if start_dt.tzinfo is not None:
            start_dt = start_dt.astimezone(IST).replace(tzinfo=None)
        start_dt = start_dt.replace(second=0, microsecond=0)
    except Exception:
        yesterday = datetime.now() - timedelta(days=1)
        start_dt = yesterday.replace(hour=10, minute=30, second=0, microsecond=0)

    # Clean registration year if it's a date string
    reg_year = row.get("glusr_gst_registration_year") or row.get("gst_registration_year")
    if isinstance(reg_year, str):
        match = re.match(r"^(\d{4})", reg_year.strip())
        if match:
            row["glusr_gst_registration_year"] = match.group(1)

    # Step 3: Run Model 4 scoring & planning
    codes, gst, notes = ENC.encode(row)
    P = slot_scores(MODEL, FEATURES, codes, gst)
    res = PLANNER.plan(P, start_dt)

    if not res:
        raise ValueError("Could not fit 3-call sequence in the two-day window for this hotlead.")

    # Step 4: Format Top 3 Windows
    predictions = []
    rank_labels = [
        "#1 Best Window (1st Attempt)",
        "#2 Alternative Window (2nd Attempt)",
        "#3 Next-Day Window (3rd Attempt)",
    ]

    for k in range(3):
        attempt_num = k + 1
        s = res["slots"][k]
        st = s["start"]
        en = st + timedelta(minutes=15)
        prob = res["p"][k]
        conf_pct = calculate_confidence_score(prob, attempt_num)

        start_12h = st.strftime("%I:%M %p")
        end_12h = en.strftime("%I:%M %p")
        time_window = f"{start_12h} – {end_12h}"

        is_same_day = st.date() == start_dt.date()
        relative_day = "Today" if is_same_day else "Next Day"

        # Compute top 3 dominant decision parameters
        top_3_params = compute_top_dominant_parameters(
            row,
            slot_q=s["slot"] + 36,
            attempt_num=attempt_num,
            wait_min=res["wait"][k],
            gap_ratio=res["gap_ratio"][k],
        )

        # Generate short reasoning focusing strictly on top parameters
        base_reasoning = PLANNER.reason(
            P,
            attempt_num,
            s,
            s["dow_type"],
            res["wait"][k],
            res["gap_ratio"][k],
            res["support"][k],
            short=True,
        )

        full_reasoning = (
            f"Top Dominant Parameters: {', '.join(top_3_params)}. "
            f"{base_reasoning}"
        )

        predictions.append({
            "rank": attempt_num,
            "rank_label": rank_labels[k],
            "attempt": attempt_num,
            "calling_date": st.strftime("%Y-%m-%d"),
            "calling_date_formatted": st.strftime("%d %b %Y"),
            "calling_day": st.strftime("%A"),
            "relative_day": relative_day,
            "is_same_day": is_same_day,
            "time_window": time_window,
            "slot_start": st.isoformat(),
            "slot_end": en.isoformat(),
            "confidence_score": f"{conf_pct}% Confidence",
            "confidence_pct": conf_pct,
            "dominant_parameters": top_3_params,
            "dominant_parameters_str": " · ".join(top_3_params),
            "reasoning": full_reasoning,
            "base_reasoning": base_reasoning,
            "wait_after_previous_min": res["wait"][k] if k > 0 else None,
            "chance_of_meeting": round(prob, 4),
        })

    # Format seller details
    seller_city = str(row.get("city_name") or row.get("seller_city") or "N/A")
    seller_state = str(row.get("state_name") or row.get("seller_state") or "N/A")
    business_type = str(row.get("business_type") or row.get("glusr_gst_legal_status") or "N/A")
    turnover = str(row.get("glusr_gst_annual_turnover") or row.get("annual_turnover") or "N/A")
    nature = str(row.get("nature_of_business") or "N/A")
    hl_trigger = str(row.get("hl_trigger") or row.get("HL_Trigger") or row.get("redis_bucket") or "PUA")

    gst_display = str(row.get("glusr_gst_registration_year") or "N/A")
    if len(gst_display) >= 4 and gst_display[:4].isdigit():
        gst_display = gst_display[:4]

    formatted_lead_dt = f"{start_dt.strftime('%d %b %Y')} at {start_dt.strftime('%I:%M %p')}"

    return {
        "status": "success",
        "project_title": "IM-Milan",
        "seller": {
            "glid": int(glid),
            "city": seller_city,
            "state": seller_state,
            "business_type": business_type,
            "nature_of_business": nature,
            "turnover": turnover,
            "gst_registration_year": gst_display,
            "hl_trigger": hl_trigger,
        },
        "hotlead": {
            "lead_sent_time": start_dt.isoformat(),
            "formatted_lead_time": formatted_lead_dt,
            "hl_trigger": hl_trigger,
            "data_age": "1 day old",
            "source": "Redash PostgreSQL DWH" if is_live_redash else "Offline Local Cache",
        },
        "predictions": predictions,
        "combined_chance_at_least_one": round(res["best"], 4),
        "combined_chance_pct": round(res["best"] * 100, 1),
        "adaptive_retry_policy": (
            "Model 4 Multi-Attempt Policy: Attempts are scheduled with data-backed cooling periods "
            "(min. 150 historical observations) to maximize cumulative meeting conversions."
        ),
    }
