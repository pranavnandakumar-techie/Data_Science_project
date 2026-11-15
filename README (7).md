# Fraud Patrol: Clean the Data, Catch the Fraudsters

**CS3006 – Introduction to Data Science · Part B: Game for learning data science concepts**

Fraud Patrol is a pygame game in which the player runs the fraud desk of a bank. A fraudster called
**the Phantom** attacks the bank with a new trick each day. To survive, the player cleans the bank's payment
data with **pandas** and finds suspicious payments with the **IQR rule** and **z-scores**.

The game never asks quiz questions. Each concept is learned by **moving a slider, watching the data change,
and living with the consequences**: missed fraud costs the bank money, and hurting honest customers costs
their trust.

---

## 1. Installation and running

**Requirements:** Python 3.9 or newer.

```bash
pip install pygame pandas numpy
python main.py
```

Run the command from the folder that contains `main.py` and `fraud_logic.py`.

### Files

| File | What it contains |
|---|---|
| `main.py` | The game: screens, ledger table, scatter plots, sliders, characters, saving results |
| `fraud_logic.py` | All the data science: data generation, pandas cleaning, IQR rule, z-scores, outcomes, explanations |
| `README.md` | This guide |
| `playtest_results.csv` | Created automatically when a player finishes the game (one row per player) |
| `playtest_runs.csv` | Created automatically when a player finishes the game (one row per attempt of each day) |

### Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pygame'` | Your virtual environment is not active, or pygame isn't installed in it. Activate it and run `pip install pygame pandas numpy`. |
| `No module named 'fraud_logic'` | You are running from the wrong folder. `cd` into the folder that contains both `.py` files. |
| The window is too big for the screen | The game window is 1280 × 740. Close other windows or lower your display scaling. |

---

## 2. How to play

### Goal
Survive all **6 days** without the bank's **Money** or customer **Trust** reaching zero. Each day is rated
with up to 3 stars (maximum 18).

### Rules
- **Missed fraud** reduces Money.
- **Hurting an honest customer** (blocking them, charging them twice, losing their payment) reduces Trust.
- Some actions are gentler: a quick check-up call to an honest customer costs only a little trust.
- If Money or Trust reaches zero, the Phantom wins.

### Each day
1. **Story:** the Phantom and Mrs. Rao (the bank manager) introduce the day's problem. Click to continue.
2. **Experiment:** move the slider(s) under **YOUR CONTROLS**.
   - The table or plot updates immediately.
   - The dark box shows the **pandas code** being run and the live numbers.
   - On plot days, **hover over any dot** to see its calculation.
   - **Show a hint** gives a clue if you are stuck.
3. **Decide:** press the green button (PROCESS PAYMENTS, REMOVE OUTLIERS or RUN THE DAY).
4. **Result:** the truth is revealed. Mrs. Rao reports what happened with real numbers, and Money/Trust change.
   - Green ring = fraud caught · Red X = fraud missed · Orange ring = honest payment hit
5. **What did I learn?** explains the concept and shows the pandas code.
6. Go to the **Next day**, or **Play this day again** with new random data. Consequences still count.

---

## 3. The six days

| Day | Phantom's trick | What you see | Your control | Concept |
|---|---|---|---|---|
| 1 · Double Trouble | Copies some payment rows | Ledger of 30 payments; copied rows in yellow | keep all rows / `drop_duplicates()` | Duplicate rows |
| 2 · The Missing Amounts | Erases some amounts (NaN) | Ledger; NaN rows in red | `dropna()` / `fillna(mean)` / `fillna(median)` / flag with `isna()` | Missing values |
| 3 · Outlier Hunt (IQR) | Hides unusually large payments | Scatter plot with Q1–Q3 band and fences | IQR multiplier m | IQR outlier rule |
| 4 · The z-score | Same kind of frauds | Scatter plot with mean and threshold lines | threshold k | z-score and threshold trade-off |
| 5 · The Whale | One ₹1.2 lakh payment | Plot squashed by one giant value | k + filter limit | Outliers distort mean and std |
| 6 · The Card Testers | Tiny ₹1–9 test payments | Plot with dots at the very bottom | k + tail (upper / lower / both) | One- and two-tailed thresholds |

---

## 4. Learning objectives

After playing, a learner should be able to:

1. **Detect and remove duplicate rows**: compare `len(df)` with `df['id'].nunique()` and use `drop_duplicates()`.
2. **Handle missing values**: explain what `isna()`, `dropna()` and `fillna()` each do, and why the choice
   changes the result (a filled value looks perfectly ordinary, a dropped row is never checked).
3. **Apply the IQR rule**: compute Q1, Q3 and IQR = Q3 − Q1, and remove values outside
   Q1 − m·IQR and Q3 + m·IQR.
4. **Compute and interpret a z-score**: z = (x − mean) / std, use the 68–95–99.7 rule, and explain the
   trade-off between false alarms (low threshold) and misses (high threshold).
5. **Recognise how one extreme value distorts the mean and standard deviation**, and fix it with a boolean
   filter such as `df[df['amount'] < limit]` before computing them.
6. **Choose an upper, lower or two-tailed rule** depending on the kind of unusual value being searched for,
   and explain why a two-tailed rule doubles the false-alarm rate (alpha) for the same k.

### What the player discovers by playing

| Day | Discovery |
|---|---|
| 1 | Keeping the copies charges customers twice; `len(df)` (30) ≠ number of unique payment IDs (24). |
| 2 | `dropna()` lets the hidden frauds pass unchecked, and `fillna(mean/median)` makes them look like ordinary payments. Only flagging with `isna()` catches them, because here the values were missing *on purpose*. |
| 3 | m = 0.5 removes many honest payments, m ≈ 1.5 removes exactly the outliers, and m = 3 lets some frauds stay. |
| 4 | A low k blocks honest customers and a high k lets fraud through: choosing k is choosing a false-alarm rate. |
| 5 | The whale pushes the std to about ₹10,000, so a ₹4,500 fraud gets z ≈ 0.15 and looks normal. After filtering out values above ₹20,000, the same payment gets z ≈ 2.7. |
| 6 | An upper-tail rule can never catch a ₹5 payment. A two-tailed rule catches both kinds, but doubles alpha. |

### Concepts and their source

| Concept | Source |
|---|---|
| z-score, 68–95–99.7 rule, upper/lower/two tails, significance level | CS3006 slides: *From sample to population, distributions and hypothesis testing* |
| Mean vs median with an outlier, boolean filtering, `drop_duplicates()`, `isna()`, `dropna()`, `fillna()` | CS3006 slides: *Pandas* |
| False alarm vs miss ("the threshold is a cost choice") | CS3006 slides: *Regression* (logistic regression, thresholds) |
| IQR rule and `quantile()` | Not in the CS3006 slides; added as an extension |

---

## 5. How it works (implementation)

- **No hard-coded answers.** Each day generates fresh random payment data with NumPy (normal payments around
  ₹2,000 plus the Phantom's frauds). The player's slider settings are applied with real pandas operations, and
  the outcome is calculated by comparing what was blocked or removed with a hidden `is_fraud` column.
- **Every replay uses new data**, so players must understand the idea rather than memorise a value.
- `fraud_logic.py` contains all the data science and no drawing code:
  - `make_day()` generates each day's data
  - `clean_duplicates()`, `eval_missing()` for days 1–2
  - `detect_iqr()` / `eval_iqr()` for day 3
  - `detect()` / `eval_z()` for days 4–6
  - `DAYS` and `DEBRIEF` hold the story text and explanations
- `main.py` contains only the interface: it is a **state machine** (`title → intro → play → result → learned
  → next day … → end`). Each screen is one method, and `step()` draws the current one 60 times per second.

---

## 6. Collecting playtest evidence

The game records evidence automatically. **Nothing needs to be downloaded**: the files appear in the game
folder on your computer.

### Procedure
1. Run `python main.py` on your laptop.
2. Ask a classmate to type **their own name** and play **until the final screen** ("THE BANK IS SAFE!" or
   "THE PHANTOM WINS..."). Do not help them. *Results are saved only when the final screen is reached; if the
   window is closed halfway, that session is not saved.*
3. Press **Play again** for the next classmate. Aim for 3–5 players.
4. After each session, ask the feedback questions below and write down the answers (or use a Google Form).
5. Open `playtest_results.csv` and `playtest_runs.csv` in Excel or Google Sheets.

### What the files contain

**`playtest_results.csv`** (one row per player)

| Column | Meaning |
|---|---|
| `player` | Name typed at the start |
| `result` | `bank safe` or `phantom wins` |
| `total_stars`, `stars_per_day` | Score overall and for each day (e.g. `D1:3 D2:0 D3:3 …`) |
| `attempts` | Total attempts including replays |
| `money_left`, `trust_left` | Final state of the bank |

**`playtest_runs.csv`** (one row per attempt)

| Column | Meaning |
|---|---|
| `player`, `day` | Who played which day |
| `settings` | The slider values used (e.g. `missing=mean`, `m=1.5`, `k=2.5; tail=both`) |
| `stars`, `money_lost`, `trust_lost` | The outcome of that attempt |
| `money_left`, `trust_left` | Bank state after the attempt |

### Using it as evidence of learning
The strongest evidence is a player **changing their settings after a bad outcome and then succeeding**. For example:

> Player A chose `missing=mean` on Day 2 (0 stars, ₹26,000 lost), replayed the day, switched to
> `missing=flag` and got 3 stars.

That sequence shows the concept was discovered through play.

### Playtest results table (fill in)

| Player | Result | Stars (/18) | Days replayed | Example of a changed decision |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

### Feedback form (fill in)

| Question | Player 1 | Player 2 | Player 3 |
|---|---|---|---|
| Name | | | |
| How fun was it? (1–5) | | | |
| In your own words: why was `fillna(mean)` a bad idea on Day 2? | | | |
| Why did the frauds look normal on Day 5 (The Whale)? | | | |
| Which day taught you the most? | | | |
| What was confusing? | | | |
| One thing you would improve | | | |

### Summary of findings (fill in after playtesting)
- Average stars:
- Days where players struggled most (most replays or lowest stars):
- Most common mistake and whether players corrected it on replay:
- Changes made to the game based on feedback:

---

## 7. Mapping to the assessment rubric

| Criterion | How Fraud Patrol addresses it |
|---|---|
| **Concept coverage and correctness** (1.5) | Six clearly named concepts (duplicates, missing values, IQR, z-score threshold, outlier distortion of mean/std, one- vs two-tailed rules). Learning objectives in Section 4; each day shows its concept and a "What did I learn?" explanation with the pandas code. |
| **Game design and learning loop** (1.5) | Goal: survive 6 days. Rules: money and trust. Feedback: instant visual update plus a computed result. Progression: a new trick each day. Player choices change the outcome, and the explanation comes only after the player has seen the consequence. |
| **Implementation** (1) | Runs in pygame without errors; all outcomes are computed with pandas/NumPy on generated data; no hard-coded answers; mouse-driven interface with sliders, hover tooltips and hints. |
| **Documentation and evidence of learning** (1) | This README (how to play, learning objectives, implementation) plus automatically recorded playtest data (`playtest_results.csv`, `playtest_runs.csv`) and the feedback forms in Section 6. |
