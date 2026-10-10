"""Turns the raw API fields into the exact category codes Model 4 was trained on."""
import re
from typing import Optional

def _norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower()

def _blank(x):
    return x is None or (isinstance(x, float) and x != x) or _norm(x) in ("", "nan", "none", "null")

class Encoder:
    def __init__(self, ref, top_cities, kept_buckets):
        self.ref = ref                      # name -> list of category labels (position = code)
        self.top_cities = set(top_cities)
        self.kept_buckets = set(kept_buckets)
        self._lut = {k: {_norm(v): i for i, v in enumerate(vals)} for k, vals in ref.items()}
        self._city_fix = {_norm(c): c for c in top_cities}
        self._bucket_fix = {_norm(c): c for c in kept_buckets}

    def code(self, col, label):
        """Code of a label; unseen labels get len(ref), which the model treats as 'missing'."""
        return self._lut[col].get(_norm(label), len(self.ref[col]))

    def encode(self, f):
        """f: dict of raw fields. Returns (codes dict, gst year or None, notes list)."""
        notes = []
        out = {}
        def plain(col, raw, name):          # business_type, turnover, nature, state: blank -> 'NA'
            if _blank(raw):
                out[col] = self.code(col, "NA"); notes.append(f"{name} was empty, treated as 'NA'")
                return
            c = self.code(col, raw)
            if c == len(self.ref[col]):
                notes.append(f"{name} '{raw}' is not a value the model was trained on, treated as missing")
            out[col] = c
        plain("business_type", f.get("business_type") or f.get("glusr_gst_legal_status"), "business_type")
        plain("annual_turnover", f.get("glusr_gst_annual_turnover") or f.get("annual_turnover"), "glusr_gst_annual_turnover")
        plain("nature_of_business", f.get("nature_of_business") or f.get("glusr_gst_nature_of_business"), "nature_of_business")
        plain("seller_state", f.get("state_name") or f.get("seller_state"), "state_name")
        city = f.get("city_name") or f.get("seller_city")                    # not in top 60 (or empty) -> 'Other'
        if _blank(city) or _norm(city) not in self._city_fix:
            out["seller_city"] = self.code("seller_city", "Other")
            notes.append("city_name is outside the top 60 cities, grouped as 'Other'" if not _blank(city) else "city_name was empty, grouped as 'Other'")
        else:
            out["seller_city"] = self.code("seller_city", self._city_fix[_norm(city)])
        trig = f.get("hl_trigger") or f.get("HL_Trigger") or f.get("redis_bucket")                   # not in the 12 kept buckets (or empty) -> 'Other'
        if _blank(trig) or _norm(trig) not in self._bucket_fix:
            out["redis_bucket"] = self.code("redis_bucket", "Other")
            notes.append(f"hl_trigger '{trig}' is outside the 12 trigger buckets, grouped as 'Other'" if not _blank(trig) else "hl_trigger was empty, grouped as 'Other'")
        else:
            out["redis_bucket"] = self.code("redis_bucket", self._bucket_fix[_norm(trig)])
        y = f.get("glusr_gst_registration_year") or f.get("gst_registration_year")
        if isinstance(y, str):
            m = re.match(r"^(\d{4})", y.strip())
            if m:
                y = m.group(1)
        try:
            y = float(y)
            if y != y or y < 1900 or y > 2100: y = None
        except (TypeError, ValueError):
            y = None
        if y is None: notes.append("glusr_gst_registration_year was empty or invalid, treated as missing")
        return out, y, notes
