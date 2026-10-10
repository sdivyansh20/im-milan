"""Scores every 15-minute slot (09:00-20:59) for the 1st, 2nd and 3rd attempt of one seller, with the Model 4 quarter-hour model."""
import numpy as np, pandas as pd

Q0, NQ = 36, 48                      # slot 36 = 09:00 ; 48 slots to 20:45
DOWS = {0: (0, 1, 2, 3, 4), 1: (5, 6)}   # day type 0 = Mon-Fri, 1 = Sat/Sun (averaged, as in the training analysis)

def slot_scores(model, features, codes, gst_year):
    """returns P[n][daytype] = array of 48 probabilities, n in 1..3"""
    rows = []
    for n in (1, 2, 3):
        for dt, dows in DOWS.items():
            for d in dows:
                for qi in range(Q0, Q0 + NQ):
                    rows.append((qi, d, n))
    X = pd.DataFrame(rows, columns=["q", "dow", "n"])
    for k, v in codes.items(): X[k] = v
    X["gst_registration_year"] = np.nan if gst_year is None else gst_year
    p = model.predict_proba(X[features])[:, 1]
    X["p"] = p
    X["dt"] = np.where(X.dow >= 5, 1, 0)
    g = X.groupby(["n", "dt", "q"]).p.mean()
    return {n: {dt: g.loc[(n, dt)].reindex(range(Q0, Q0 + NQ)).to_numpy() for dt in (0, 1)} for n in (1, 2, 3)}
