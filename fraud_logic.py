"""
fraud_logic.py - the data and the rules behind Fraud Patrol (no drawing code here).

Days 1-2: DATA CLEANING on a small ledger table
    Day 1  duplicates      -> df.drop_duplicates()
    Day 2  missing values  -> dropna() / fillna(mean) / fillna(median) / isna() flag
Day 3: OUTLIERS WITH THE IQR RULE
    Day 3  remove values outside [Q1 - m*IQR, Q3 + m*IQR]
Days 4-6: Z-SCORES on a scatter plot
    Day 4  z = (amount - mean) / std, block if z > k
    Day 5  one huge outlier drags the mean and std -> filter it out first: df[df['amount'] < limit]
    Day 6  tiny payments are LOW outliers -> upper / lower / both tails
"""
import math

import numpy as np
import pandas as pd

START_MONEY = 100_000          # Rs the bank can afford to lose to fraud
START_TRUST = 100              # customer trust, %
TRUST_ANGRY = 4                # % trust lost per honest customer who is wrongly hurt
TRUST_CALL = 1                 # % trust lost per honest customer who just gets a check-up call
TESTER_LOSS = 15_000           # a missed card tester comes back and steals this much
NO_LIMIT = 150                 # filter slider value (Rs thousands) meaning "no filter"

NAMES = ["Asha", "Ravi", "Meena", "Arjun", "Priya", "Karthik", "Divya", "Rahul", "Sneha", "Vikram",
         "Anita", "Deepak", "Fatima", "Kiran", "Lakshmi", "Manoj", "Nisha", "Om", "Pooja", "Rohan",
         "Sanjay", "Tara", "Usha", "Varun", "Yamini", "Zoya", "Bala", "Chitra", "Gopal", "Hema"]


def alpha(k, tail):
    """Share of a normal curve beyond the threshold (like a significance level)."""
    one = 0.5 * math.erfc(k / math.sqrt(2))
    return 2 * one if tail == "both" else one


# =================================================================== data
def _ledger(rng, n):
    amt = np.clip(rng.normal(2000, 450, n), 200, None).round(0)
    return pd.DataFrame({"customer": rng.choice(NAMES, n), "amount": amt, "is_fraud": False, "kind": "normal"})


def _normal(rng, n):
    return [{"amount": a, "is_fraud": False, "kind": "normal"}
            for a in np.clip(rng.normal(2000, 450, n), 200, None).round(0)]


def _fraud(rng, n, lo, hi, kind="fraud"):
    return [{"amount": float(round(a)), "is_fraud": True, "kind": kind} for a in rng.uniform(lo, hi, n)]


def make_day(day, seed=None):
    rng = np.random.default_rng(seed)
    if day == 1:
        df = _ledger(rng, 24)
        df.insert(0, "payment_id", np.arange(5001, 5025))
        dup = df.sample(6, random_state=int(rng.integers(1e9)))      # exact copies of 6 rows
        df = pd.concat([df, dup]).sort_values("payment_id", kind="stable").reset_index(drop=True)
        return df
    if day == 2:
        df = _ledger(rng, 30)
        df.insert(0, "payment_id", np.arange(6001, 6031))
        idx = rng.choice(30, 6, replace=False)
        df["true_amount"] = df["amount"]
        df.loc[idx[:4], ["is_fraud", "kind"]] = [True, "missing_fraud"]
        df.loc[idx[:4], "true_amount"] = rng.uniform(5000, 9000, 4).round(0)
        df.loc[idx[4:], "kind"] = "missing_honest"
        df.loc[idx, "amount"] = np.nan
        return df
    rows = []
    if day == 3:
        rows = _normal(rng, 120) + _fraud(rng, 6, 3600, 6000)
    elif day == 4:
        rows = _normal(rng, 120) + _fraud(rng, 6, 4300, 6500)
    elif day == 5:
        rows = _normal(rng, 120) + _fraud(rng, 6, 4300, 6500) + _fraud(rng, 1, 110_000, 130_000, "whale")
    elif day == 6:
        rows = _normal(rng, 200) + _fraud(rng, 2, 4300, 6500) + _fraud(rng, 4, 1, 9, "tester")
    df = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    df.insert(0, "payment_id", np.arange(7001, 7001 + len(df)))
    return df


# =================================================================== day 1: duplicates
def clean_duplicates(df, choice):
    return df.drop_duplicates() if choice == "drop" else df


def eval_duplicates(df, choice):
    out = clean_duplicates(df, choice)
    twice = int(out["payment_id"].duplicated().sum())
    return {"rows": len(out), "double_charged": twice, "loss": 0, "trust_loss": TRUST_ANGRY * twice,
            "lines": ([f"{twice} customers were charged TWICE because their payment row appeared twice. "
                       "They are furious!"] if twice else
                      ["Every customer was charged exactly once. Clean ledger!"]),
            "stars": 3 if twice == 0 else 0}


# =================================================================== day 2: missing values
def eval_missing(df, choice):
    miss = df[df["amount"].isna()]
    frauds = miss[miss["is_fraud"]]
    honest = miss[~miss["is_fraud"]]
    nf, nh = len(frauds), len(honest)
    stolen = float(frauds["true_amount"].sum())
    if choice == "dropna":
        lines = [f"dropna() removed {len(miss)} rows, so those payments were never checked.",
                 f"{nf} of them were fraud and went straight through: Rs {stolen:,.0f} stolen.",
                 f"{nh} honest customers' payments silently disappeared. They are angry."]
        res = dict(loss=stolen, trust_loss=TRUST_ANGRY * nh, caught=0)
    elif choice in ("mean", "median"):
        fill = df["amount"].mean() if choice == "mean" else df["amount"].median()
        lines = [f"fillna({choice}) wrote Rs {fill:,.0f} into all {len(miss)} empty amounts.",
                 f"Every filled payment now looks perfectly ordinary, so the {nf} frauds passed: Rs {stolen:,.0f} stolen.",
                 f"{nh} honest customers were charged a made-up amount instead of their real one. They are angry."]
        res = dict(loss=stolen, trust_loss=TRUST_ANGRY * nh, caught=0)
    else:  # flag with isna()
        lines = [f"isna() flagged all {len(miss)} payments with a missing amount for review.",
                 f"All {nf} frauds were caught! {nh} honest customers just got a quick check-up call."]
        res = dict(loss=0.0, trust_loss=TRUST_CALL * nh, caught=nf)
    res.update(lines=lines, n_missing=len(miss), n_fraud=nf, n_honest=nh,
               stars=3 if choice == "flag" else 0)
    return res


# =================================================================== days 3-5: z-scores
def detect(df, k, limit=NO_LIMIT, tail="upper"):
    """z = (amount - mean) / std, where mean/std come only from payments below the filter limit."""
    d = df.copy()
    used = d["amount"] < limit * 1000 if limit < NO_LIMIT else pd.Series(True, index=d.index)
    d["used"] = used
    mean = float(d.loc[used, "amount"].mean())
    std = float(d.loc[used, "amount"].std())
    d["z"] = (d["amount"] - mean) / std
    if tail == "upper":
        d["blocked"] = d["z"] > k
    elif tail == "lower":
        d["blocked"] = d["z"] < -k
    else:
        d["blocked"] = d["z"].abs() > k
    return d, mean, std


def detect_iqr(df, m):
    """IQR rule: anything outside [Q1 - m*IQR, Q3 + m*IQR] is an outlier and gets removed."""
    d = df.copy()
    q1, q3 = d["amount"].quantile(0.25), d["amount"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - m * iqr, q3 + m * iqr
    d["blocked"] = (d["amount"] < lo) | (d["amount"] > hi)
    return d, float(q1), float(q3), float(iqr), float(lo), float(hi)


def eval_iqr(df, m):
    return _score(detect_iqr(df, m)[0], verb="removed")


def eval_z(df, k, limit=NO_LIMIT, tail="upper"):
    return _score(detect(df, k, limit, tail)[0], verb="blocked")


def _score(d, verb):
    fraud = d["is_fraud"]
    caught = d[fraud & d["blocked"]]
    missed = d[fraud & ~d["blocked"]]
    false = d[~fraud & d["blocked"]]

    def cost(rows):
        return float(sum(TESTER_LOSS if r["kind"] == "tester" else r["amount"] for _, r in rows.iterrows()))

    n = int(fraud.sum())
    lines = [f"You caught {len(caught)} of {n} frauds."]
    if len(missed):
        extra = ""
        if (missed["kind"] == "tester").any():
            extra = f" ({int((missed['kind'] == 'tester').sum())} card testers came back later and stole Rs {TESTER_LOSS:,} each)"
        lines.append(f"{len(missed)} slipped through: Rs {cost(missed):,.0f} lost{extra}.")
    if len(false):
        lines.append(f"You {verb} {len(false)} honest customers' payments. They are furious.")
    res = dict(caught=len(caught), n_fraud=n, missed=len(missed), false_blocks=len(false),
               loss=cost(missed), trust_loss=TRUST_ANGRY * len(false), lines=lines)
    if res["missed"] == 0 and res["false_blocks"] <= 1:
        res["stars"] = 3
    elif res["caught"] >= 0.7 * n and res["false_blocks"] <= 4:
        res["stars"] = 2
    elif res["caught"] >= 0.4 * n:
        res["stars"] = 1
    else:
        res["stars"] = 0
    return res


# =================================================================== the days
DAYS = {
    1: dict(title="Day 1: Double Trouble", kind="table", concept="Duplicate rows: drop_duplicates()",
            controls=["duplicates"], start={"duplicates": "keep"}, run="PROCESS PAYMENTS",
            intro=[("phantom", "Hehehe, I'm THE PHANTOM. I copied some rows in your payment ledger. Pay them all!"),
                   ("manager", "Look at the ledger. Rows highlighted in YELLOW have the same Payment ID as the row "
                               "above, so the same payment was logged twice. Use the switch, then PROCESS PAYMENTS.")],
            hint="Count the rows vs the different Payment IDs. A payment ID should appear only once."),
    2: dict(title="Day 2: The Missing Amounts", kind="table", concept="Missing values: dropna, fillna, isna",
            controls=["missing"], start={"missing": "dropna"}, run="PROCESS PAYMENTS",
            intro=[("phantom", "I erased the amounts on some payments. NaN, NaN, NaN!"),
                   ("manager", "Rows highlighted in RED have a missing amount (NaN). Choose what to do with them: "
                               "drop them, fill them with the mean or median, or flag them with isna(). Each choice "
                               "has a different result. Try them!")],
            hint="Ask yourself WHY an amount would go missing. Who benefits if nobody looks at these rows?"),
    3: dict(title="Day 3: Outlier Hunt (IQR)", kind="iqr", concept="Outliers with the IQR rule (box plot fences)",
            controls=["m"], start={"m": 0.5}, run="REMOVE OUTLIERS",
            intro=[("phantom", "My fraud payments are hiding among your honest ones. They're bigger than usual... a bit."),
                   ("manager", "The shaded band is the middle 50% of payments: from Q1 (25th percentile) to Q3 "
                               "(75th percentile). Its height is the IQR = Q3 - Q1."),
                   ("manager", "The red lines are the fences: Q1 - m x IQR and Q3 + m x IQR. Anything outside a fence is an "
                               "OUTLIER and gets removed. Move m: too small removes honest payments, too big lets "
                               "fraud stay.")],
            hint="Most textbooks use m = 1.5. Try it, then try 0.5 and 3 and watch how many dots turn red."),
    4: dict(title="Day 4: The z-score", kind="plot", concept="z = (amount - mean) / std, and the threshold k",
            controls=["k"], start={"k": 0.5}, run="RUN THE DAY",
            intro=[("manager", "Now a second way to spot unusual payments. Each dot is a payment. The z-score says how many "
                               "standard deviations a payment is above the average:  z = (amount - mean) / std."),
                   ("manager", "Any payment with z > k is BLOCKED (red dot). Move k: too low and honest customers "
                               "get blocked; too high and fraud slips through. Hover over a dot to see its z.")],
            hint="About 95% of normal payments have z between -2 and +2. Frauds are far above the crowd."),
    5: dict(title="Day 5: The Whale", kind="plot", concept="One huge outlier drags the mean and std: filter first",
            controls=["k", "limit"], start={"k": 2.5, "limit": NO_LIMIT}, run="RUN THE DAY",
            intro=[("phantom", "Meet my BIG friend: one Rs 1,20,000 payment. Watch your mean and std explode..."),
                   ("manager", "Look at the mean and std in the bar below the plot. New slider: a FILTER. Only payments "
                               "below the limit are used to calculate the mean and std, like df[df['amount'] < limit].")],
            hint="With no filter, how big is the std? Drag the filter down and watch the std and the red line."),
    6: dict(title="Day 6: The Card Testers", kind="plot", concept="Upper, lower and two-tailed thresholds",
            controls=["k", "tail"], start={"k": 2.5, "tail": "upper"}, run="RUN THE DAY",
            intro=[("phantom", "My team tests stolen cards with TINY payments of Rs 1-9. You only ever look UP!"),
                   ("manager", "Tiny payments have a big NEGATIVE z. New slider: the TAIL. Upper blocks z > k, lower "
                               "blocks z < -k, both blocks |z| > k.")],
            hint="The testers are at the very bottom of the plot. Which tail points down?"),
}

DEBRIEF = {
    3: ("The IQR rule finds outliers using quartiles",
        "Q1 and Q3 cut off the lowest and highest 25% of values, so the IQR = Q3 - Q1 is the width of the middle 50%. "
        "Values beyond Q1 - 1.5 x IQR or Q3 + 1.5 x IQR are called outliers (these are the whiskers of a box plot). "
        "A small multiplier removes honest values too; a large one lets real outliers stay. Because quartiles only "
        "depend on the middle of the data, a few extreme values hardly move the fences.",
        "q1, q3 = df['amount'].quantile(0.25), df['amount'].quantile(0.75)\niqr = q3 - q1\n"
        "df = df[(df['amount'] >= q1 - 1.5*iqr) & (df['amount'] <= q3 + 1.5*iqr)]"),
    1: ("Duplicates double-count everything",
        "When the same row appears twice, the bank processes it twice, and every total, mean and count includes it "
        "twice. df.drop_duplicates() keeps only the first copy of each identical row. Always compare the number of "
        "rows with the number of unique IDs.",
        "len(df), df['payment_id'].nunique()\ndf = df.drop_duplicates()"),
    2: ("Missing values: what you do with them matters",
        "isna() finds the gaps. dropna() deletes those rows, so nobody checks them. fillna(mean) or fillna(median) "
        "replaces them with an ordinary value, so they LOOK normal even when they're not. Here the amounts were missing "
        "on purpose, so the safest choice was to flag the rows with isna() and check them.",
        "df['amount'].isna().sum()\ndf['needs_review'] = df['amount'].isna()"),
    4: ("The threshold is a trade-off",
        "z counts how many standard deviations a payment is from the mean. For a normal curve about 68% of values have "
        "|z| < 1, 95% have |z| < 2 and 99.7% have |z| < 3. A LOW k blocks honest customers (false alarms); a HIGH k "
        "lets fraud through (misses).",
        "mean, std = df['amount'].mean(), df['amount'].std()\ndf['z'] = (df['amount'] - mean) / std\n"
        "df['blocked'] = df['z'] > k"),
    5: ("One giant outlier hides the others",
        "The whale pulls the mean up and makes the std huge, so ordinary frauds end up with a small z and look normal. "
        "Filtering out the extreme value before computing the mean and std fixes this. Clean first, then measure.",
        "clean = df[df['amount'] < 20000]\nmean, std = clean['amount'].mean(), clean['amount'].std()\n"
        "df['z'] = (df['amount'] - mean) / std"),
    6: ("Outliers can be LOW too",
        "Big frauds are in the upper tail; tiny card-test payments are in the lower tail. An upper-tail rule can never "
        "catch them. A two-tailed rule (|z| > k) catches both, but with the same k it also flags twice as many honest "
        "payments (alpha doubles).",
        "upper:  df[df['z'] > k]\nlower:  df[df['z'] < -k]\nboth:   df[df['z'].abs() > k]"),
}
