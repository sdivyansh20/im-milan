import json, os, time, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union
import joblib
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from .encode import Encoder
from .scoring import slot_scores
from .planner import Planner

ART = Path(os.environ.get("BTC_ARTIFACTS", Path(__file__).resolve().parent.parent / "artifacts"))
MODEL_VERSION = "model4-q15-gap-2026-10"
IST = timezone(timedelta(hours=5, minutes=30))
log = logging.getLogger("btc")

bundle = joblib.load(ART / "model4_q15.joblib")
meta = json.loads((ART / "meta.json").read_text())
MODEL, FEATURES = bundle["model"], bundle["features"]
ENC = Encoder(bundle["ref"], meta["top_cities"], meta["buckets_kept"])
PLANNER = Planner(meta)

app = FastAPI(title="Best Time to Call - Model 4", version=MODEL_VERSION)

class PlanRequest(BaseModel):
    fk_glusr_usr_id: Union[int, str]
    lead_sent_time: str = Field(description="Hotlead creation time in IST, e.g. 2026-10-12T14:07:00. A value with a time zone is converted to IST.")
    business_type: Optional[str] = None
    glusr_gst_annual_turnover: Optional[str] = None
    nature_of_business: Optional[str] = None
    glusr_gst_registration_year: Optional[Union[int, float, str]] = None
    city_name: Optional[str] = None
    state_name: Optional[str] = None
    hl_trigger: Optional[str] = None

def parse_ist(s):
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00").replace(" ", "T", 1) if "T" not in s else s.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "lead_sent_time must look like 2026-10-12T14:07:00 (IST)")
    if d.tzinfo is not None: d = d.astimezone(IST).replace(tzinfo=None)
    return d.replace(second=0, microsecond=0)

def hm(m): return f"{m//60:02d}:{m%60:02d}"

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/categories")
def categories():
    """What the model knows. Use it to check that your database values match."""
    r = bundle["ref"]
    return {"business_type": r["business_type"], "glusr_gst_annual_turnover": r["annual_turnover"],
            "nature_of_business": r["nature_of_business"], "state_name": r["seller_state"],
            "city_name_top60": meta["top_cities"], "hl_trigger_kept": meta["buckets_kept"],
            "rule": "city_name and hl_trigger outside these lists, or empty, are grouped as 'Other'"}

@app.post("/plan")
def plan(req: PlanRequest):
    t0 = time.perf_counter()
    start = parse_ist(req.lead_sent_time)
    codes, gst, notes = ENC.encode(req.model_dump())
    P = slot_scores(MODEL, FEATURES, codes, gst)
    res = PLANNER.plan(P, start)
    out = {"fk_glusr_usr_id": req.fk_glusr_usr_id, "lead_sent_time_ist": start.isoformat(timespec="minutes"),
           "model_version": MODEL_VERSION, "window": "hotlead creation day and the next day, IST", "notes": notes}
    if res is None:
        out["plan"] = None; out["message"] = "No three-call plan fits in these two days."
    else:
        rows, run = [], 1.0
        for k in range(3):
            n = k + 1; s = res["slots"][k]; run *= 1 - res["p"][k]
            st = s["start"]; en = st + timedelta(minutes=15)
            rows.append({"attempt": n, "slot_start": st.isoformat(timespec="minutes") + "+05:30", "slot_end": en.isoformat(timespec="minutes") + "+05:30",
                         "date": st.date().isoformat(), "time": f"{hm(st.hour*60+st.minute)}-{hm(en.hour*60+en.minute)}",
                         "chance_of_meeting": round(res["p"][k], 4), "chance_at_least_one_so_far": round(1 - run, 4),
                         "wait_after_previous_min": res["wait"][k] if k else None,
                         "calls_behind_this_wait": res["support"][k] if k else None,
                         "reason": PLANNER.reason(P, n, s, s["dow_type"], res["wait"][k], res["gap_ratio"][k], res["support"][k])})
        out.update({"plan": rows, "combined_chance_at_least_one": round(res["best"], 4), "plans_tied_within_1pct": res["ties"],
                    "thin_data": not res["strict"]})
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("plan glid=%s lead=%s ms=%s", req.fk_glusr_usr_id, out["lead_sent_time_ist"], out["ms"])
    return out
