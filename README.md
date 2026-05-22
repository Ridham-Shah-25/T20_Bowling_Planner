# Batter Intel - Match-Day Bowling Planner

Batter Intel is a Streamlit application that generates a six-ball bowling plan for a selected batter and match situation. The app uses historical ball-by-ball cricket data to recommend what type of delivery to bowl, why it was selected, what variation to use, and which fielding zones are most likely to be involved.

The app is designed for T20 match-day planning. It combines interpretable statistical methods with a lightweight LLM narration layer. All delivery decisions are made by deterministic Python logic; the LLM only explains the already-selected plan in plain English.

---

## What The App Does

For a selected batter and current match state, the app:

1. Filters historical ball-by-ball data to the relevant batter, bowler type, bowler arm, phase, and optional ground.
2. Builds a composite delivery table using length, line, and bowling angle.
3. Applies recency weighting so newer seasons influence the recommendation more than older seasons.
4. Applies smoothing so small samples do not dominate the ranking.
5. Determines whether the bowling intent should be `CONTAIN` or `WICKET`.
6. Scores each delivery type for the selected intent.
7. Adjusts recommendations using selected bowler execution data when a bowler is selected.
8. Uses similar batter data when the selected batter has limited historical data.
9. Mines historical setup-ball sequences that have wicket lift.
10. Generates a six-ball over plan.
11. Displays clear reasoning for each ball, including raw records, smoothed ranking signals, variation choice, fielding zones, and danger-zone warnings.

---

## Core Idea

A bowling plan should not simply ask:

```text
What has worked overall?
```

It should ask:

```text
Against this batter, in this match situation, against this bowler type, what delivery is most useful and reliable?
```

The app therefore combines:

- batter-specific historical evidence,
- context-specific filtering,
- recency weighting,
- smoothing,
- similar batter supplementation,
- bowler execution adjustment,
- sequence pattern mining,
- and intent-based scoring.

---

## Inputs Used By The App

The app uses the following user inputs:

- Batter
- Bowler type
- Bowler arm
- Optional bowler for direct matchup and execution data
- Stadium / ground
- Innings
- Over number
- Team runs
- Wickets down
- Batter runs
- Balls faced
- Runs needed, if chasing
- Balls left, if chasing

The over number determines the phase:

```text
Overs 1-6   -> Powerplay
Overs 7-16  -> Middle
Overs 17-20 -> Death
```

In a chase, balls left is inferred from the selected over but can still be edited for partial overs or unusual match situations.

---

## Delivery Fingerprint

The main unit of analysis is a composite delivery type:

```text
length + line + bowling angle
```

For example:

```text
LengthBall + OutsideOff + Over
Yorker + MiddleStump + Round
BackOfALength + DownLeg + Over
```

The app evaluates this combination together instead of evaluating length, line, or angle independently.

Variation is handled separately after the main length and line are selected. This keeps the primary sample size larger and more reliable.

---

## Recency Weighting

Older cricket data may be less relevant than recent data. To account for this, the app gives newer seasons more influence.

The current weighting logic is:

```text
latest season weight = 1.00
one year older       = 0.85
two years older      = 0.85 x 0.85
minimum weight       = 0.30
```

Example with respect to the current year 2026:

```text
2026 ball -> 1.00 effective ball
2025 ball -> 0.85 effective balls
2024 ball -> 0.72 effective balls
2023 ball -> 0.61 effective balls
2022 ball -> 0.52 effective balls
2021 ball -> 0.44 effective balls
2020 ball -> 0.38 effective balls
2019 ball -> 0.32 effective balls
```

This creates the concept of effective balls.

```text
raw balls       = actual balls in the data
effective balls = balls after recency weighting
```

Raw balls are shown in many UI tables because they are easy to understand. Effective balls are used in scoring confidence and smoothing.

---

## Smoothing

Raw percentages can be misleading when sample sizes are small.

Example:

```text
1 wicket in 1 ball = 100% wicket rate
```

That does not mean the delivery is truly a 100% wicket option.

To prevent this, the app uses empirical-Bayes style smoothing. It blends the delivery's own result with a broader prior.

General formula:

```text
smoothed rate =
(delivery success count + prior rate x prior balls)
/
(delivery effective balls + prior balls)
```

For runs:

```text
smoothed runs/ball =
(delivery runs + prior runs/ball x prior balls)
/
(delivery effective balls + prior balls)
```

The app smooths:

- Wicket%
- Dot%
- Boundary%
- Contain%
- Runs per ball

Current prior strengths:

```text
Dot prior balls      = 24
Boundary prior balls = 24
Contain prior balls  = 24
Runs prior balls     = 24
Wicket prior balls   = 48
```

Wickets use a stronger prior because wickets are rarer and noisier.

---

## Reliability Rules

A delivery can enter the six-ball plan if it has:

```text
raw balls >= 6
OR
raw wickets >= 2
```

The table reliability marker uses effective balls:

```text
effective balls >= 6
OR
raw wickets >= 2
```

This means:

- the app can generate plans from understandable raw evidence,
- but scoring still penalizes older or weaker evidence through effective balls.

---

## Intent Engine

The app first determines the over intent:

```text
CONTAIN
or
WICKET
```

The intent depends on:

- innings,
- phase,
- runs needed,
- balls left,
- team current run rate,
- batter scoring rate,
- wickets down,
- and ground scoring context when available.

Examples:

- In a chase during death overs, the app usually prioritizes containment.
- In a powerplay where the batting side is below par, the app may look for wickets.
- In middle overs, the default is usually pressure-building containment.

---

## Delivery Scoring

Each delivery type receives a score based on the selected intent.

### CONTAIN Score

The app rewards:

- high smoothed contain rate,
- low scoring rate,
- some wicket value,
- and reliable sample size.

Conceptually:

```text
CONTAIN score =
0.80 x smoothed contained-ball%
+ 0.20 x low-SR value
+ half wicket bonus
```

Then the score is multiplied by confidence:

```text
confidence = min(1.0, effective balls / 6)
```

### WICKET Score

The app rewards:

- high smoothed wicket rate,
- high smoothed dot rate,
- low scoring rate,
- wicket evidence,
- and reliable sample size.

Conceptually:

```text
WICKET score =
0.60 x smoothed wicket%
+ 0.20 x smoothed dot%
+ 0.20 x low-SR value
+ wicket bonus
```

Again, the final score is confidence-weighted.

The UI shows score breakdowns so users can see why the top delivery types rank highly.

---

## Similar Batter Model

When the selected batter has limited data, the app supplements with similar batters.

This is not based only on overall strike rate.

The app builds delivery-vulnerability profiles using:

```text
phase
bowler type
bowler arm
length
line
bowling angle
```

For each context, it compares:

```text
smoothed runs/ball delta
smoothed dot-rate delta
smoothed boundary-rate delta
smoothed wicket-rate delta
```

Delta means the batter's result against that delivery context compared with that batter's own baseline.

The app then uses cosine similarity to find batters with similar vulnerability patterns.

This helps answer:

```text
Which batters react similarly to the same delivery types?
```

This is useful when the selected batter's own sample is thin.

---

## Sequence Lift Mining

Some wickets come from setup patterns, not isolated balls.

The app mines historical 2-ball and 3-ball patterns inside the same over.

It compares:

```text
wicket rate after setup sequence
vs
normal wicket rate of the final delivery
```

Formula:

```text
lift = sequence wicket rate / final-delivery baseline wicket rate
```

Example:

```text
Yorker MiddleStump alone wicket rate = 4%
LengthBall OutsideOff -> Yorker MiddleStump wicket rate = 8%

lift = 8 / 4 = 2.0x
```

A sequence is considered only if it has:

```text
minimum attempts >= 8
minimum wickets >= 2
minimum lift >= 1.25x
```

This is a sequential pattern mining and lift analysis technique. It is used because the cricket question is sequential:

```text
Did the previous ball make this wicket ball more effective?
```

Raw wicket counts alone cannot answer that.

---

## Bowler Execution Adjustment

A delivery can be good against a batter but poor for the selected bowler.

For example:

```text
Yorker OutsideOff may be strong against the batter,
but if the selected bowler often misses that ball,
the recommendation should be reduced.
```

When a specific bowler is selected, the app builds a bowler execution profile by length + line + angle.

It measures:

```text
bowler dot%
bowler boundary%
bowler wicket%
bowler contain%
bowler error%
bowler runs/ball
```

These rates are smoothed toward the broader bowler-type context to avoid overreacting to small samples.

Bowler execution can adjust the final delivery score, but its influence is capped:

```text
maximum bowler execution influence = 30%
```

This keeps the recommendation centered on batter matchup evidence while still accounting for whether the selected bowler can execute the ball.

---

## Variation Recommendation

The primary recommendation is the main delivery type:

```text
length + line + bowling angle
```

After that is selected, the app looks for the best variation within the chosen length and line.

Example:

```text
LengthBall + OutsideOff
```

Possible variations may include:

```text
NoMovement
InSwinger
OffCutter
SlowerBall
```

The variation recommendation uses historical variation-level stats such as:

- wicket rate,
- dot-ball rate,
- SR conceded,
- and sample size.

This is shown separately in the ball card so users know the difference between:

```text
why this length + line was chosen
```

and:

```text
why this variation was chosen
```

---

## Fielding Zones

The app shows likely fielding zones based on historical fielding positions for the selected delivery type.

These are not strict field-setting commands.

They mean:

```text
where the ball has often gone historically for this delivery type
```

The UI labels this as:

```text
Likely fielding zones
```

---

## Danger Zone

The app identifies a danger-zone delivery:

```text
the delivery type where the batter has scored most freely
```

The danger zone is shown using raw numbers only:

```text
raw SR
raw balls
```

The plan avoids this delivery when possible.

---

## Six-Ball Plan Generation

Once the delivery table is built and scored, the app creates a six-ball plan.

The plan builder:

1. Filters out delivery types without enough raw evidence.
2. Scores every eligible delivery.
3. Prefers reliable deliveries.
4. Avoids the danger-zone delivery.
5. Applies a maximum repeat cap so the same delivery is not overused.
6. Uses sequence patterns if a strong setup pattern exists.
7. Selects the best variation within each chosen length and line.
8. Returns a deterministic six-ball over plan.

The repeat cap prevents the plan from becoming unrealistic by recommending the same ball six times.

---

## LLM Narration

The LLM does not decide the plan.

The deterministic Python engine chooses:

- length,
- line,
- angle,
- variation,
- intent,
- likely fielding zones,
- and danger zone.

The LLM is only asked to write a short explanation for each already-selected ball.

This keeps the app reproducible:

```text
same data + same inputs = same bowling plan
```

---

## UI Explanation Layers

The app includes explanation panels for:

- how the app works,
- data science methods used,
- recency weighting,
- reliability rules,
- smoothing formulas,
- data used for the recommendation,
- score breakdown,
- similar batter supplement,
- sequence evidence,
- bowler execution adjustment,
- danger zone,
- and glossary terms.

The goal is that a user can understand not only what the app recommends, but also why the recommendation was made.

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Groq API for narration
- Deterministic statistical recommendation engine

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app expects the required ball-by-ball data source to be available through the configured app environment.

---

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## Important Design Principle

This project intentionally prioritizes interpretability over black-box modelling.

The app does not try to predict cricket outcomes with a hidden model. Instead, it uses transparent statistical reasoning:

```text
filter relevant data
weight recent evidence
smooth noisy samples
score by match intent
adjust for bowler execution
explain the plan clearly
```

That makes the recommendations easier to inspect, challenge, and trust.
