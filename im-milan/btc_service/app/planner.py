"""Picks the 1st, 2nd and 3rd call slot within two days of the hotlead, each after the one before.
Wait between calls comes from past calls: the model-based ratio R(n, daytype, hour, wait) from the gap tables."""
from datetime import datetime, timedelta
import numpy as np

SUP_MIN = 150          # a wait is only allowed if at least this many past calls had about this wait
TIE = 0.99             # plans within 1% of the best are treated as equal; the earliest is shown
GD, NG = 15, 192

class Planner:
    def __init__(self, meta):
        g = meta["gap"]
        self.R = {n: np.array(g["R"][str(n)], dtype=float) for n in (2, 3)}      # [daytype][hour 0..11][192]
        self.S = {n: np.array(g["S"][str(n)], dtype=float) for n in (2, 3)}
        self.prior = {n: np.array(meta["prior"][str(n)], dtype=float) for n in (1, 2, 3)}   # [96]: 48 weekday + 48 weekend

    def plan(self, P, start):
        """P[n][dt] -> 48 probs. start: naive datetime in IST."""
        cut = start.hour * 60 + start.minute
        t, we, hr, q, day, p = [], [], [], [], [], {1: [], 2: [], 3: []}
        for i in range(2):
            d = (start + timedelta(days=i)).date(); w = 1 if d.weekday() >= 5 else 0
            for k in range(48):
                m = 540 + k * 15
                if i == 0 and m < cut: continue
                t.append(i * 1440 + m); we.append(w); hr.append(k >> 2); q.append(k); day.append(d)
                for n in (1, 2, 3): p[n].append(P[n][w][k])
        K = len(t)
        if K < 3: return None
        t = np.array(t); we = np.array(we); hr = np.array(hr)
        p = {n: np.array(v) for n, v in p.items()}
        D = t[None, :] - t[:, None]
        idx = D // GD - 1
        ok = (D > 0) & (idx >= 0) & (idx < NG)
        ic = np.clip(idx, 0, NG - 1)
        G = {n: self.R[n][we[None, :], hr[None, :], ic] for n in (2, 3)}   # G[n][x,y]: ratio for slot y reached from slot x
        sup = {n: self.S[n][ic] for n in (2, 3)}
        for strict in (True, False):
            m2 = ok & ((sup[2] >= SUP_MIN) if strict else True)
            m3 = ok & ((sup[3] >= SUP_MIN) if strict else True)
            P2 = p[2][None, :] * G[2]; P3 = p[3][None, :] * G[3]
            V = 1 - (1 - p[1][:, None, None]) * (1 - P2[:, :, None]) * (1 - P3[None, :, :])
            V = np.where(m2[:, :, None] & m3[None, :, :], V, -1.0)
            best = V.max()
            if best < 0: continue
            cand = V >= best * TIE
            ties = int(cand.sum())
            a, b, c = np.unravel_index(int(np.argmax(cand.ravel())), V.shape)   # earliest (a,b,c)
            ids = (int(a), int(b), int(c))
            slots = [dict(start=start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=int(t[j] // 1440), minutes=int(t[j] % 1440)),
                          dow_type=int(we[j]), slot=int(q[j]), t=int(t[j])) for j in ids]
            return dict(slots=slots, ids=ids, best=float(V[a, b, c]), ties=ties, strict=strict,
                        p=[float(p[1][a]), float(P2[a, b]), float(P3[b, c])],
                        gap_ratio=[1.0, float(G[2][a, b]), float(G[3][b, c])],
                        wait=[0, int(D[a, b]), int(D[b, c])],
                        support=[0, int(sup[2][a, b]), int(sup[3][b, c])],
                        base=[float(p[1][a]), float(p[2][b]), float(p[3][c])])
        return None

    # --- plain-language reason for each row (concise, highlighting highest impact parameters) ---
    def reason(self, P, n, slot, dt, wait, ratio, support, short=True):
        k = slot["slot"]
        avg = float(np.mean(P[n][dt])); me = float(P[n][dt][k])
        pri = self.prior[n][(48 if dt else 0) + k]; pavg = float(np.mean(self.prior[n][(48 if dt else 0):(48 if dt else 0) + 48]))
        rel = me / avg - 1; typ = pri / pavg - 1; adj = (me / avg) / (pri / pavg) - 1
        dn = "weekend" if dt else "weekday"; ordn = ["1st", "2nd", "3rd"][n - 1]

        if short:
            if n == 1:
                drivers = []
                if typ >= 0.03:
                    drivers.append(f"Peak general window (+{round(typ*100)}% uplift)")
                elif typ <= -0.03:
                    drivers.append(f"General window ({round(typ*100)}% vs avg)")
                if adj >= 0.03:
                    drivers.append(f"Seller profile affinity (+{round(adj*100)}% boost)")
                elif adj <= -0.03:
                    drivers.append(f"Seller profile impact ({round(adj*100)}%)")
                if not drivers:
                    drivers.append(f"High-conversion slot ({round(rel*100):+d}% vs daytime avg)")
                return "Key drivers: " + " and ".join(drivers[:2]) + "."
            else:
                e = ratio - 1
                w = (f"{wait/60:.1f}".rstrip("0").rstrip(".") + " hours") if wait >= 120 else ("1 hour" if wait == 60 else f"{wait} mins")
                cmp = f"{abs(round(e*100))}% {'better' if e > 0 else 'worse'}" if abs(round(e*100)) >= 2 else "stable conversion"
                drivers = [f"Optimal {w} cooldown ({cmp} across {support:,} past calls)"]
                if typ >= 0.05:
                    drivers.append(f"peak {ordn}-call window (+{round(typ*100)}%)")
                return "Key driver: " + "; ".join(drivers[:2]) + "."

        parts = []
        if rel >= .03: parts.append(f"{round(rel*100)}% above this seller's average {dn} slot for a {ordn} call")
        elif rel <= -.03: parts.append(f"{round(-rel*100)}% below this seller's average {dn} slot for a {ordn} call")
        else: parts.append(f"close to this seller's average {dn} slot for a {ordn} call")
        if typ >= .03: parts.append(f"for sellers in general this slot is {round(typ*100)}% above the average")
        elif typ <= -.03: parts.append(f"for sellers in general this slot is {round(-typ*100)}% below the average")
        if adj >= .03: parts.append(f"this seller's profile adds a further {round(adj*100)}%")
        elif adj <= -.03: parts.append(f"this seller's profile takes off {round(-adj*100)}%")
        if n > 1:
            e = ratio - 1
            w = (f"{wait/60:.1f}".rstrip("0").rstrip(".")+" hours") if wait >= 120 else ("1 hour" if wait == 60 else f"{wait} minutes")
            cmp = "about the same as" if abs(round(e * 100)) < 2 else f"{abs(round(e*100))}% {'better' if e > 0 else 'worse'} than"
            parts.append(f"{w} after the {'1st' if n == 2 else '2nd'} attempt: in the history, a {ordn} call after about this wait converted {cmp} a typical {ordn} call ({support:,} similar calls)")
        else:
            parts.append("part of the plan with the best combined chance of a meeting")
        return "; ".join(parts[:4]) + "."
