"""
Batter Intel — Match-day Bowling Planner
=========================================
Architecture
------------
1. All delivery decisions are made by deterministic Python code.
2. Every delivery is treated as a COMPOSITE FINGERPRINT:
       length + line + variation (bowlingDetailId) + side (bowlingFromId) + hand (bowlingHandId)
   These five dimensions are NEVER evaluated independently.
3. The LLM (Groq) is called ONLY to write one plain-English explanation
   sentence per ball, given pre-decided numbers. It cannot change any decision.
4. Same data + same match situation = identical output, every time.
5. When batter-specific data is thin, similar batsmen (by statistical profile)
   are used to supplement — clearly flagged in the UI.
6. Wicket-taking sequences mined from historical data inform setup/surprise
   ball placement within the over.
"""

import streamlit as st
import pandas as pd
import json
import re
import html
from collections import Counter
from groq import Groq
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Batter Intel",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { max-width: 800px; padding: 2rem 1.5rem; }

.stat-row { display:flex; gap:8px; flex-wrap:wrap; margin:0.7rem 0; }
.chip { background:#f3f4f6; border-radius:8px; padding:5px 12px;
        font-size:0.79rem; font-weight:500; color:#111; }
.chip span { color:#888; font-weight:400; margin-right:3px; }

.ball-card { border:1.5px solid #e5e7eb; border-radius:12px;
             padding:14px 16px; margin:8px 0; background:#fff; }
.ball-num  { font-size:0.68rem; font-weight:600; color:#9ca3af;
             text-transform:uppercase; letter-spacing:.07em; margin-bottom:4px; }
.delivery  { font-size:1.05rem; font-weight:700; color:#111; margin-bottom:2px; }
.sub-delivery { font-size:0.82rem; color:#6b7280; margin-bottom:8px; }
.why-text  { font-size:0.82rem; color:#374151; line-height:1.6; margin-bottom:9px; }

.tag { font-size:0.74rem; border-radius:6px; padding:3px 9px;
       display:inline-block; font-weight:500; margin:2px 2px 2px 0; }
.t-field   { background:#eff6ff; color:#1d4ed8; }
.t-wicket  { background:#fef2f2; color:#b91c1c; }
.t-contain { background:#f0fdf4; color:#15803d; }
.t-setup   { background:#faf5ff; color:#7e22ce; }
.t-surprise{ background:#fff7ed; color:#c2410c; }
.t-warn    { background:#fffbeb; color:#92400e; }

.insight-box { background:#fffbeb; border-left:3px solid #f59e0b;
               border-radius:0 8px 8px 0; padding:10px 14px;
               font-size:0.83rem; color:#78350f; margin:0.8rem 0; line-height:1.6; }
.danger-box  { background:#fef2f2; border-left:3px solid #ef4444;
               border-radius:0 8px 8px 0; padding:10px 14px;
               font-size:0.83rem; color:#7f1d1d; margin:0.8rem 0; line-height:1.6; }
.similar-box { background:#eff6ff; border-left:3px solid #3b82f6;
               border-radius:0 8px 8px 0; padding:10px 14px;
               font-size:0.83rem; color:#1e3a5f; margin:0.8rem 0; line-height:1.6; }
.warn-box    { background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px;
               padding:8px 12px; font-size:0.76rem; color:#475569; margin:6px 0; }

.htable { font-size:0.77rem; width:100%; border-collapse:collapse; }
.htable th { font-weight:600; color:#6b7280; text-align:left;
             padding:5px 7px; border-bottom:1px solid #e5e7eb; }
.htable td { padding:5px 7px; color:#111; border-bottom:1px solid #f9fafb; }
.g { color:#15803d; font-weight:600; }
.r { color:#b91c1c; font-weight:600; }

.hdr { display:flex; align-items:baseline; gap:10px; margin-bottom:.3rem; flex-wrap:wrap; }
.bname { font-size:1.3rem; font-weight:700; color:#111; }
.ctag  { font-size:0.73rem; background:#111; color:#fff;
         border-radius:20px; padding:3px 10px; font-weight:500; }
hr.div { border:none; border-top:1px solid #f0f0f0; margin:1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MIN_RELIABLE         = 6   # minimum balls for a delivery fingerprint to be trusted
MIN_FIELD_OBS        = 3    # minimum times a fielding position must appear to be listed
MIN_SIMILAR_THRESH   = 30   # if fewer balls than this, supplement with similar batsmen
MIN_PROFILE_BALLS    = 30   # minimum balls to build a batter profile
MIN_VULNERABILITY_GROUP_BALLS = 3  # minimum balls for one batter delivery-response feature
MAX_REPEATS          = 3    # max times the same fingerprint can be bowled in one over (within-over adaptation cap)
MIN_SEQUENCE_ATTEMPTS = 5   # minimum total attempts for a setup -> final-ball sequence
MIN_SEQUENCE_WICKETS  = 2   # minimum wickets after a setup sequence
MIN_SEQUENCE_LIFT     = 1.25  # setup must beat final-ball baseline by this multiplier
DOT_PRIOR_BALLS      = 24   # prior strength for dot-ball smoothing
BOUNDARY_PRIOR_BALLS = 24   # prior strength for boundary-risk smoothing
CONTAIN_PRIOR_BALLS  = 24   # prior strength for contain-ball smoothing
RUNS_PRIOR_BALLS     = 24   # prior strength for expected-runs smoothing
WICKET_PRIOR_BALLS   = 48   # stronger prior because wickets are rare/noisy
BOWLER_PRIOR_BALLS   = 36   # prior strength for bowler execution smoothing
BOWLER_EXECUTION_WEIGHT = 0.30  # max influence of bowler execution on final score
PARTIAL_MATCHUP_MIN_TOTAL_BALLS = 6
PARTIAL_MATCHUP_MIN_DELIVERY_BALLS = 3
PARTIAL_MATCHUP_MIN_DELIVERY_WICKETS = 1
PARTIAL_MATCHUP_MAX_WEIGHT = 0.20
PARTIAL_MATCHUP_CONFIDENCE_BALLS = 30
PARTIAL_MATCHUP_SR_DIFF = 25
PARTIAL_MATCHUP_DOT_DIFF = 12
PARTIAL_MATCHUP_BOUNDARY_DIFF = 8
RECENCY_DECAY_PER_YEAR = 0.85  # each older year keeps this fraction of influence
MIN_RECENCY_WEIGHT = 0.30      # old data still contributes as fallback evidence
DATA_YEAR_COL = "_data_year"
DATA_WEIGHT_COL = "_data_weight"

# Composite delivery key columns — evaluated TOGETHER, never independently
# bowlingDetailId (variation) is intentionally excluded: it is determined via
# a secondary lookup *within* the chosen line+length so fingerprints have
# larger, more reliable sample sizes.
DELIVERY_COLS = ["lengthTypeId", "lineTypeId", "bowlingFromId"]

def filter_by_bowler_type_hand(df: pd.DataFrame,
                                bowler_type: str,
                                bowler_hand: str) -> pd.DataFrame:
    """
    Filter dataframe to rows matching the selected bowler type and hand.
    Both values are matched directly against the data — no mapping applied.
    Falls back to unfiltered if the filter would leave fewer than 10 rows.
    """
    filtered = df.copy()

    # Filter by bowling hand (Right/Left) — direct match
    if bowler_hand and "bowlingHandId" in filtered.columns:
        mask = filtered["bowlingHandId"].str.lower() == bowler_hand.lower()
        if mask.sum() >= 10:
            filtered = filtered[mask]

    # Filter by bowling type — direct match against actual data values
    if bowler_type and "bowlingTypeId" in filtered.columns:
        mask = filtered["bowlingTypeId"].str.lower() == bowler_type.lower()
        if mask.sum() >= 10:
            filtered = filtered[mask]

    return filtered


def _clean_option_list(series: pd.Series) -> list[str]:
    return sorted(
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )


def bowler_type_options_for(df: pd.DataFrame, bowler: str = "") -> list[str]:
    if "bowlingTypeId" not in df.columns:
        return ["Fast"]

    source = df.copy()
    if bowler and "bowlingPlayer" in source.columns:
        source = source[source["bowlingPlayer"].astype(str) == str(bowler)]

    return _clean_option_list(source["bowlingTypeId"]) or ["Fast"]


def bowler_hand_options_for(
    df: pd.DataFrame,
    bowler: str = "",
    bowler_type: str = "",
) -> list[str]:
    if "bowlingHandId" not in df.columns:
        return ["Right", "Left"]

    source = df.copy()
    if bowler and "bowlingPlayer" in source.columns:
        source = source[source["bowlingPlayer"].astype(str) == str(bowler)]
    if bowler_type and "bowlingTypeId" in source.columns:
        source = source[
            source["bowlingTypeId"].astype(str).str.strip() == str(bowler_type)
        ]

    return _clean_option_list(source["bowlingHandId"]) or ["Right", "Left"]


def bowler_options_for(
    df: pd.DataFrame,
    bowler_type: str = "",
    bowler_hand: str = "",
) -> list[str]:
    if "bowlingPlayer" not in df.columns:
        return []

    source = df.copy()
    if bowler_type and "bowlingTypeId" in source.columns:
        source = source[
            source["bowlingTypeId"].astype(str).str.strip() == str(bowler_type)
        ]
    if bowler_hand and "bowlingHandId" in source.columns:
        source = source[
            source["bowlingHandId"].astype(str).str.strip() == str(bowler_hand)
        ]

    return _clean_option_list(source["bowlingPlayer"])


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["batruns", "overNumber", "control", "shotAngle", "shotMagnitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["isWicket", "bat_out", "isWide", "isNoBall"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    df["inningNumber"] = df["inningNumber"].astype(str).str.strip()
    for c in DELIVERY_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    # Ensure ballNumber is numeric for sequence mining
    if "ballNumber" in df.columns:
        df["ballNumber"] = pd.to_numeric(df["ballNumber"], errors="coerce").fillna(0)
    df = add_recency_weights(df)
    return df


def load_data(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, low_memory=False)
    return prepare_data(df)


@st.cache_data(show_spinner="Loading ball-by-ball dataset from Hugging Face...")
def load_huggingface_data(repo_id: str, filename: str, token: str) -> pd.DataFrame:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_csv(local_path, low_memory=False)
    return prepare_data(df)


def clear_plan_state():
    for key in ["plan", "narration", "plan_batter", "plan_ctx", "similar_info", "plan_signature"]:
        st.session_state.pop(key, None)


def is_out(row) -> bool:
    return (row.get("isWicket", "false") in ("true", "1") or
            row.get("bat_out",  "false") in ("true", "1"))


def phase_of(over: float) -> str:
    if over <= 6:  return "Powerplay"
    if over <= 16: return "Middle"
    return "Death"


def sr(runs, balls):
    return round(runs / balls * 100, 1) if balls else 0.0

def pct(a, b):
    return round(a / b * 100, 1) if b else 0.0


def infer_data_years(df: pd.DataFrame) -> pd.Series:
    if "year" not in df.columns:
        return pd.Series(pd.NA, index=df.index)

    years = pd.to_numeric(df["year"], errors="coerce")
    if years.notna().sum() > 0:
        return years

    extracted = df["year"].astype(str).str.extract(r"(\d{4})", expand=False)
    years = pd.to_numeric(extracted, errors="coerce")
    if years.notna().sum() > 0:
        return years

    return pd.Series(pd.NA, index=df.index)


def add_recency_weights(df: pd.DataFrame) -> pd.DataFrame:
    years = infer_data_years(df)
    df[DATA_YEAR_COL] = years

    if years.notna().sum() == 0:
        df[DATA_WEIGHT_COL] = 1.0
        return df

    latest_year = int(years.max())
    age = latest_year - years
    weights = RECENCY_DECAY_PER_YEAR ** age
    df[DATA_WEIGHT_COL] = weights.clip(lower=MIN_RECENCY_WEIGHT).fillna(1.0)
    return df


def row_weights(df: pd.DataFrame) -> pd.Series:
    if DATA_WEIGHT_COL not in df.columns:
        return pd.Series(1.0, index=df.index)
    return pd.to_numeric(df[DATA_WEIGHT_COL], errors="coerce").fillna(1.0)


def smooth_rate(successes, balls, prior_rate, prior_strength):
    total = balls + prior_strength
    return (successes + prior_rate * prior_strength) / total if total else 0.0


def smooth_runs(runs, balls, prior_runs_per_ball, prior_strength):
    total = balls + prior_strength
    return (runs + prior_runs_per_ball * prior_strength) / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Legal-delivery helpers
# ─────────────────────────────────────────────────────────────────────────────
def _legal_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Return legal delivery rows for rate calculations."""
    out = df.copy()
    if "isWide" in out.columns:
        out = out[~out["isWide"].isin(["true", "1"])]
    if "isNoBall" in out.columns:
        out = out[~out["isNoBall"].isin(["true", "1"])]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Similar batsman profiling (delivery-vulnerability based; no control column)
# ─────────────────────────────────────────────────────────────────────────────
def build_batter_profiles(df_all: pd.DataFrame) -> dict:
    """
    Build batter profiles for similarity matching.
    Similarity is based on delivery-vulnerability features: how a batter's
    outcomes change against specific phase + bowler type/hand + delivery
    fingerprints. Broad batting-style stats are retained for display/fallback.
    """
    profiles = {}
    legal = df_all[
        ~df_all.get("isWide", pd.Series(["false"] * len(df_all))).isin(["true", "1"])
    ].copy()
    if legal.empty or "battingPlayer" not in legal.columns:
        return profiles

    legal["phase"] = legal["overNumber"].apply(phase_of)

    for batter, g_valid in legal.groupby("battingPlayer"):
        actual_balls = len(g_valid)
        if actual_balls < MIN_PROFILE_BALLS:
            continue

        gw = row_weights(g_valid)
        b = float(gw.sum())
        if b <= 0:
            continue

        r = float((g_valid["batruns"] * gw).sum())
        wicket_mask = g_valid.apply(is_out, axis=1)
        w = float(gw[wicket_mask].sum())
        dots = float(gw[g_valid["batruns"] == 0].sum())
        boundaries = float(gw[g_valid["batruns"].isin([4, 6])].sum())

        overall_runs_per_ball = r / b if b else 0.0
        overall_dot_rate = dots / b if b else 0.0
        overall_boundary_rate = boundaries / b if b else 0.0
        overall_wicket_rate = w / b if b else 0.0

        pp    = g_valid[g_valid["phase"] == "Powerplay"]
        mid   = g_valid[g_valid["phase"] == "Middle"]
        death = g_valid[g_valid["phase"] == "Death"]

        def weighted_phase_sr(sub: pd.DataFrame) -> float:
            sw = row_weights(sub)
            return sr(float((sub["batruns"] * sw).sum()), float(sw.sum()))

        def weighted_phase_dot(sub: pd.DataFrame) -> float:
            sw = row_weights(sub)
            return pct(float(sw[sub["batruns"] == 0].sum()), float(sw.sum()))

        features = {}
        group_cols = [
            c for c in [
                "phase", "bowlingTypeId", "bowlingHandId", *DELIVERY_COLS
            ]
            if c in g_valid.columns
        ]

        g_features = g_valid.copy()
        for c in group_cols:
            g_features[c] = g_features[c].fillna("Unknown").astype(str).str.strip()
            g_features.loc[g_features[c].isin(["", "nan", "None"]), c] = "Unknown"

        for key, dg in g_features.groupby(group_cols):
            actual_group_balls = len(dg)
            if actual_group_balls < MIN_VULNERABILITY_GROUP_BALLS:
                continue

            dgw = row_weights(dg)
            balls = float(dgw.sum())
            if balls <= 0:
                continue

            dg_runs = float((dg["batruns"] * dgw).sum())
            dg_dots = float(dgw[dg["batruns"] == 0].sum())
            dg_boundaries = float(dgw[dg["batruns"].isin([4, 6])].sum())
            dg_wickets = float(dgw[dg.apply(is_out, axis=1)].sum())

            sm_rpb = smooth_runs(dg_runs, balls, overall_runs_per_ball, RUNS_PRIOR_BALLS)
            sm_dot = smooth_rate(dg_dots, balls, overall_dot_rate, DOT_PRIOR_BALLS)
            sm_boundary = smooth_rate(
                dg_boundaries, balls, overall_boundary_rate, BOUNDARY_PRIOR_BALLS
            )
            sm_wicket = smooth_rate(
                dg_wickets, balls, overall_wicket_rate, WICKET_PRIOR_BALLS
            )

            key_text = "|".join(map(str, key if isinstance(key, tuple) else (key,)))
            features[f"{key_text}|rpb_delta"] = (sm_rpb - overall_runs_per_ball) * 100
            features[f"{key_text}|dot_delta"] = (sm_dot - overall_dot_rate) * 100
            features[f"{key_text}|boundary_delta"] = (
                sm_boundary - overall_boundary_rate
            ) * 100
            features[f"{key_text}|wicket_delta"] = (
                sm_wicket - overall_wicket_rate
            ) * 100

        profiles[batter] = {
            "sr":           sr(r, b),
            "boundary_pct": pct(boundaries, b),
            "dot_pct":      pct(dots, b),
            "wkt_freq":     pct(w, b),
            "pp_sr":        weighted_phase_sr(pp)    if len(pp) >= 10    else sr(r, b),
            "mid_sr":       weighted_phase_sr(mid)   if len(mid) >= 10   else sr(r, b),
            "death_sr":     weighted_phase_sr(death) if len(death) >= 10 else sr(r, b),
            "pp_dot_pct":   weighted_phase_dot(pp)       if len(pp) >= 10    else pct(dots, b),
            "death_dot_pct":weighted_phase_dot(death)    if len(death) >= 10 else pct(dots, b),
            "_features": features,
            "_balls": actual_balls,
            "_effective_balls": round(b, 1),
            "_similarity_basis": "delivery vulnerability",
        }
    return profiles


def find_similar_batsmen(
    target: str,
    profiles: dict,
    n: int = 5,
    phase: str | None = None,
    bowler_type: str | None = None,
    bowler_hand: str | None = None,
) -> list:
    """
    Find top-n batters with similar delivery vulnerabilities.
    Falls back to broad batting-style distance when delivery-feature overlap is thin.
    """
    if target not in profiles:
        return []

    context_prefix = None
    if phase and bowler_type and bowler_hand:
        context_prefix = f"{phase}|{bowler_type}|{bowler_hand}|"

    def filtered_features(features: dict) -> dict:
        if not context_prefix:
            return features
        filtered = {k: v for k, v in features.items() if k.startswith(context_prefix)}
        return filtered if len(filtered) >= 4 else features

    def cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if len(common) < 4:
            return 0.0
        dot = sum(float(a[k]) * float(b[k]) for k in common)
        norm_a = sum(float(a[k]) ** 2 for k in common) ** 0.5
        norm_b = sum(float(b[k]) ** 2 for k in common) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    target_features = filtered_features(profiles[target].get("_features", {}))
    similarities = []
    for name, prof in profiles.items():
        if name == target:
            continue
        sim = cosine(target_features, filtered_features(prof.get("_features", {})))
        if sim > 0:
            similarities.append((name, round(sim, 4), prof))

    if similarities:
        similarities.sort(key=lambda x: -x[1])
        return similarities[:n]

    broad_keys = [
        "sr", "boundary_pct", "dot_pct", "wkt_freq",
        "pp_sr", "mid_sr", "death_sr", "pp_dot_pct", "death_dot_pct",
    ]
    t = profiles[target]
    all_vals = {
        k: [p[k] for p in profiles.values() if k in p]
        for k in broad_keys
    }
    mins = {k: min(v) for k, v in all_vals.items() if v}
    maxs = {k: max(v) for k, v in all_vals.items() if v}
    ranges = {
        k: (maxs[k] - mins[k]) if maxs[k] != mins[k] else 1.0
        for k in mins
    }

    def norm(val, key):
        return (val - mins[key]) / ranges[key]

    distances = []
    for name, p in profiles.items():
        if name == target:
            continue
        usable_keys = [k for k in ranges if k in t and k in p]
        if not usable_keys:
            continue
        dist = sum((norm(t[k], k) - norm(p[k], k)) ** 2 for k in usable_keys) ** 0.5
        distances.append((name, round(dist, 4), p))

    distances.sort(key=lambda x: x[1])
    return distances[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence mining — find setup patterns with wicket lift
# ─────────────────────────────────────────────────────────────────────────────
def mine_wicket_sequences(df: pd.DataFrame) -> list:
    """
    Find 2-3 ball setup sequences whose wicket rate beats the final delivery's
    normal wicket rate. This avoids promoting sequences that only look good
    because the final ball is already a strong wicket-taking delivery.
    """
    available_cols = [c for c in DELIVERY_COLS if c in df.columns]

    if not available_cols:
        return []
    if "fixtureId" not in df.columns or "overNumber" not in df.columns:
        return []
    if "ballNumber" not in df.columns:
        return []

    legal = _legal_balls(df)
    if legal.empty:
        return []

    legal_weights = row_weights(legal)
    prior_balls = float(legal_weights.sum())
    prior_wickets = float(legal_weights[legal.apply(is_out, axis=1)].sum())
    prior_wicket_rate = prior_wickets / prior_balls if prior_balls else 0.04

    final_attempts = Counter()
    final_wickets = Counter()
    sequence_attempts = Counter()
    sequence_wickets = Counter()
    sequence_raw_attempts = Counter()
    sequence_raw_wickets = Counter()
    sequence_examples = {}

    def invalid_fp(fp):
        return any(str(v).strip().lower() in ("unknown", "nan", "none", "") for v in fp)

    group_cols = ["fixtureId", "overNumber"]
    if "inningNumber" in legal.columns:
        group_cols.insert(1, "inningNumber")
    if "battingPlayer" in legal.columns:
        group_cols.append("battingPlayer")

    def consecutive_balls(rows: list[dict]) -> bool:
        nums = [float(r.get("ballNumber", 0)) for r in rows]
        return all((nums[j] - nums[j - 1]) == 1 for j in range(1, len(nums)))

    for _, over_df in legal.groupby(group_cols):
        over_df = over_df.sort_values("ballNumber")
        balls = over_df.to_dict("records")
        phase = phase_of(float(over_df["overNumber"].iloc[0]))

        for i, ball in enumerate(balls):
            final_fp = _fp_tuple(ball, available_cols)
            if invalid_fp(final_fp):
                continue

            ball_weight = float(ball.get(DATA_WEIGHT_COL, 1.0) or 1.0)
            final_key = (phase, final_fp)
            final_attempts[final_key] += ball_weight
            if is_out(ball):
                final_wickets[final_key] += ball_weight

            for setup_len in (1, 2):
                if i - setup_len < 0:
                    continue
                setup_rows = balls[i - setup_len:i]
                sequence_rows = setup_rows + [ball]
                if not consecutive_balls(sequence_rows):
                    continue
                setup_fp = tuple(_fp_tuple(s, available_cols) for s in setup_rows)
                if any(invalid_fp(fp) for fp in setup_fp):
                    continue

                seq_key = (phase, setup_fp, final_fp)
                sequence_attempts[seq_key] += ball_weight
                sequence_raw_attempts[seq_key] += 1
                if is_out(ball):
                    sequence_wickets[seq_key] += ball_weight
                    sequence_raw_wickets[seq_key] += 1

                sequence_examples[seq_key] = {
                    "setup": [
                        {c: str(s.get(c, "Unknown")) for c in available_cols}
                        for s in setup_rows
                    ],
                    "wicket": {
                        c: str(ball.get(c, "Unknown")) for c in available_cols
                    },
                    "phase": phase,
                }

    sequences = []
    for seq_key, attempts in sequence_attempts.items():
        phase, setup_fp, final_fp = seq_key
        wickets = sequence_wickets[seq_key]
        raw_attempts = sequence_raw_attempts[seq_key]
        raw_wickets = sequence_raw_wickets[seq_key]
        if raw_attempts < MIN_SEQUENCE_ATTEMPTS or raw_wickets < MIN_SEQUENCE_WICKETS:
            continue

        final_key = (phase, final_fp)
        final_balls = final_attempts[final_key]
        final_wkts = final_wickets[final_key]
        seq_rate = smooth_rate(
            wickets, attempts, prior_wicket_rate, WICKET_PRIOR_BALLS
        )
        final_rate = smooth_rate(
            final_wkts, final_balls, prior_wicket_rate, WICKET_PRIOR_BALLS
        )
        if final_rate <= 0:
            continue

        lift = seq_rate / final_rate
        if lift < MIN_SEQUENCE_LIFT:
            continue

        item = sequence_examples[seq_key].copy()
        item.update({
            "attempts": int(raw_attempts),
            "effective_attempts": round(float(attempts), 1),
            "wickets": int(raw_wickets),
            "sequence_wicket_rate": round(seq_rate * 100, 1),
            "final_wicket_rate": round(final_rate * 100, 1),
            "lift": round(lift, 2),
        })
        sequences.append(item)

    sequences.sort(key=lambda s: (-s["lift"], -s["attempts"]))
    return sequences


def _fp_tuple(row_or_dict, cols):
    """Convert a row/dict to a hashable fingerprint tuple."""
    return tuple(str(row_or_dict.get(c, "")) for c in cols)


# ─────────────────────────────────────────────────────────────────────────────
# Core: build composite delivery fingerprint table
# ─────────────────────────────────────────────────────────────────────────────
def build_delivery_table(
    df: pd.DataFrame,
    prior_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Groups every ball by its full composite delivery fingerprint:
        length + line + variation + side + hand

    If prior_df is supplied, delivery-level smoothing first uses the same
    delivery fingerprint from prior_df as the prior. This lets phase-specific
    samples borrow strength from all-phase evidence for the same line/length.

    Returns one row per unique fingerprint with:
        balls, runs, sr, wkt_pct, dot_pct, boundary_pct,
        top_fielding_positions (list), reliable (bool)

    This is the ONLY table the decision engine uses.
    """
    df = df[~df.get("isWide", pd.Series(["false"]*len(df))).isin(["true", "1"])].copy()
    if df.empty:
        return pd.DataFrame()

    available_cols = [c for c in DELIVERY_COLS if c in df.columns]
    weights = row_weights(df)
    prior_balls = float(weights.sum())
    prior_runs = float((df["batruns"] * weights).sum())
    prior_wickets = float(weights[df.apply(is_out, axis=1)].sum())
    prior_dots = float(weights[df["batruns"] == 0].sum())
    prior_boundaries = float(weights[df["batruns"].isin([4, 6])].sum())
    prior_contain = float(weights[df["batruns"].isin([0, 1, 2])].sum())
    prior_dot_rate = prior_dots / prior_balls if prior_balls else 0.0
    prior_boundary_rate = prior_boundaries / prior_balls if prior_balls else 0.0
    prior_wicket_rate = prior_wickets / prior_balls if prior_balls else 0.0
    prior_contain_rate = prior_contain / prior_balls if prior_balls else 0.0
    prior_runs_per_ball = prior_runs / prior_balls if prior_balls else 0.0

    delivery_prior_lookup = {}
    if prior_df is not None and not prior_df.empty:
        prior_source = prior_df[
            ~prior_df.get("isWide", pd.Series(["false"] * len(prior_df))).isin(["true", "1"])
        ].copy()
        prior_cols = [c for c in available_cols if c in prior_source.columns]

        if len(prior_cols) == len(available_cols):
            for pkeys, pg in prior_source.groupby(available_cols, dropna=False):
                if not isinstance(pkeys, tuple):
                    pkeys = (pkeys,)
                if any(str(k).strip().lower() in ("unknown", "nan", "none", "") for k in pkeys):
                    continue

                pgw = row_weights(pg)
                p_effective_balls = float(pgw.sum())
                if p_effective_balls <= 0:
                    continue

                p_raw_wickets = int(pg.apply(is_out, axis=1).sum())
                if p_effective_balls < MIN_RELIABLE and p_raw_wickets < 2:
                    continue

                p_weighted_runs = float((pg["batruns"] * pgw).sum())
                p_weighted_wickets = float(pgw[pg.apply(is_out, axis=1)].sum())
                p_weighted_dots = float(pgw[pg["batruns"] == 0].sum())
                p_weighted_boundaries = float(pgw[pg["batruns"].isin([4, 6])].sum())
                p_weighted_contain = float(pgw[pg["batruns"].isin([0, 1, 2])].sum())

                delivery_prior_lookup[tuple(str(k) for k in pkeys)] = {
                    "runs_per_ball": p_weighted_runs / p_effective_balls,
                    "wicket_rate": p_weighted_wickets / p_effective_balls,
                    "dot_rate": p_weighted_dots / p_effective_balls,
                    "boundary_rate": p_weighted_boundaries / p_effective_balls,
                    "contain_rate": p_weighted_contain / p_effective_balls,
                    "effective_balls": round(p_effective_balls, 1),
                }

    records = []
    for keys, g in df.groupby(available_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if any(str(k).strip().lower() in ("unknown", "nan", "none", "") for k in keys):
            continue

        b = len(g)
        gw = row_weights(g)
        effective_balls = float(gw.sum())
        if effective_balls <= 0:
            continue

        r = int(g["batruns"].sum())
        raw_w = int(g.apply(is_out, axis=1).sum())
        weighted_runs = float((g["batruns"] * gw).sum())
        weighted_wickets = float(gw[g.apply(is_out, axis=1)].sum())
        weighted_dots = float(gw[g["batruns"] == 0].sum())
        weighted_boundaries = float(gw[g["batruns"].isin([4, 6])].sum())
        weighted_contain = float(gw[g["batruns"].isin([0, 1, 2])].sum())
        key_tuple = tuple(str(k) for k in keys)
        delivery_prior = delivery_prior_lookup.get(key_tuple, {})

        row_prior_dot_rate = delivery_prior.get("dot_rate", prior_dot_rate)
        row_prior_boundary_rate = delivery_prior.get("boundary_rate", prior_boundary_rate)
        row_prior_wicket_rate = delivery_prior.get("wicket_rate", prior_wicket_rate)
        row_prior_contain_rate = delivery_prior.get("contain_rate", prior_contain_rate)
        row_prior_runs_per_ball = delivery_prior.get("runs_per_ball", prior_runs_per_ball)

        sm_dot = smooth_rate(weighted_dots, effective_balls, row_prior_dot_rate, DOT_PRIOR_BALLS)
        sm_bdy = smooth_rate(weighted_boundaries, effective_balls, row_prior_boundary_rate, BOUNDARY_PRIOR_BALLS)
        sm_wkt = smooth_rate(weighted_wickets, effective_balls, row_prior_wicket_rate, WICKET_PRIOR_BALLS)
        sm_contain = smooth_rate(weighted_contain, effective_balls, row_prior_contain_rate, CONTAIN_PRIOR_BALLS)
        sm_rpb = smooth_runs(weighted_runs, effective_balls, row_prior_runs_per_ball, RUNS_PRIOR_BALLS)
        fp_series = g["fieldingPosition"].dropna() if "fieldingPosition" in g.columns \
                    else pd.Series([], dtype=str)
        fp_series = fp_series.astype(str)
        fp_counts = fp_series[fp_series.str.strip() != ""].value_counts()
        top_fields = fp_counts[fp_counts >= MIN_FIELD_OBS].index.tolist()[:5]

        row = {c_name: str(k) for c_name, k in zip(available_cols, keys)}
        row.update({
            "balls":         b,
            "effective_balls": round(effective_balls, 1),
            "runs":          r,
            "sr":            sr(weighted_runs, effective_balls),
            "wkt_pct":       pct(weighted_wickets, effective_balls),
            "dot_pct":       pct(weighted_dots, effective_balls),
            "boundary_pct":  pct(weighted_boundaries, effective_balls),
            "contained_ball_pct": pct(weighted_contain, effective_balls),
            "smoothed_wkt_pct": pct(sm_wkt, 1),
            "smoothed_dot_pct": pct(sm_dot, 1),
            "smoothed_boundary_pct": pct(sm_bdy, 1),
            "smoothed_contained_ball_pct": pct(sm_contain, 1),
            "smoothed_runs_per_ball": round(sm_rpb, 3),
            "smoothed_sr": round(sm_rpb * 100, 1),
            "wickets":       raw_w,
            "weighted_wickets": round(weighted_wickets, 2),
            "top_fields":    top_fields,
            "reliable":      effective_balls >= MIN_RELIABLE or raw_w >= 2,
        })
        records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def build_bowler_execution_table(df_all: pd.DataFrame,
                                 bowler: str,
                                 bowler_type: str,
                                 bowler_hand: str) -> pd.DataFrame:
    """
    Build bowler-side execution stats by delivery fingerprint.
    This answers whether the selected bowler reliably executes a delivery,
    independent of the selected batter's weakness profile.
    """
    if not bowler or "bowlingPlayer" not in df_all.columns:
        return pd.DataFrame()

    required = {"batruns", *DELIVERY_COLS}
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        return pd.DataFrame()

    df_context = filter_by_bowler_type_hand(df_all.copy(), bowler_type, bowler_hand)
    legal_context = _legal_balls(df_context)
    if legal_context.empty:
        return pd.DataFrame()

    context_weights = row_weights(legal_context)
    prior_balls = float(context_weights.sum())
    prior_runs = float((legal_context["batruns"] * context_weights).sum())
    prior_wickets = float(context_weights[legal_context.apply(is_out, axis=1)].sum())
    prior_dots = float(context_weights[legal_context["batruns"] == 0].sum())
    prior_boundaries = float(context_weights[legal_context["batruns"].isin([4, 6])].sum())
    prior_contain = float(context_weights[legal_context["batruns"].isin([0, 1, 2])].sum())
    prior_dot_rate = prior_dots / prior_balls if prior_balls else 0.0
    prior_boundary_rate = prior_boundaries / prior_balls if prior_balls else 0.0
    prior_wicket_rate = prior_wickets / prior_balls if prior_balls else 0.0
    prior_contain_rate = prior_contain / prior_balls if prior_balls else 0.0
    prior_runs_per_ball = prior_runs / prior_balls if prior_balls else 0.0

    df_bowler_all = df_context[df_context["bowlingPlayer"].astype(str) == str(bowler)].copy()
    if df_bowler_all.empty:
        return pd.DataFrame()
    legal_bowler = _legal_balls(df_bowler_all)
    if legal_bowler.empty:
        return pd.DataFrame()

    legal_cols = [c for c in DELIVERY_COLS if c in legal_bowler.columns]
    records = []
    for keys, g in legal_bowler.groupby(legal_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if any(str(k).strip().lower() in ("unknown", "nan", "none", "") for k in keys):
            continue

        b = len(g)
        gw = row_weights(g)
        effective_balls = float(gw.sum())
        if effective_balls <= 0:
            continue

        r = int(g["batruns"].sum())
        weighted_runs = float((g["batruns"] * gw).sum())
        weighted_wickets = float(gw[g.apply(is_out, axis=1)].sum())
        weighted_dots = float(gw[g["batruns"] == 0].sum())
        weighted_boundaries = float(gw[g["batruns"].isin([4, 6])].sum())
        weighted_contain = float(gw[g["batruns"].isin([0, 1, 2])].sum())

        error_mask = pd.Series(False, index=df_bowler_all.index)
        if "isWide" in df_bowler_all.columns:
            error_mask = error_mask | df_bowler_all["isWide"].isin(["true", "1"])
        if "isNoBall" in df_bowler_all.columns:
            error_mask = error_mask | df_bowler_all["isNoBall"].isin(["true", "1"])
        err_df = df_bowler_all[error_mask]
        err_count = 0
        weighted_err_count = 0.0
        if not err_df.empty:
            err_sub = err_df.copy()
            for col, val in zip(legal_cols, keys):
                err_sub = err_sub[err_sub[col].astype(str) == str(val)]
            err_count = len(err_sub)
            weighted_err_count = float(row_weights(err_sub).sum())
        total_attempts = b + err_count
        weighted_total_attempts = effective_balls + weighted_err_count
        context_error_rate = 0.0
        if "isWide" in df_context.columns or "isNoBall" in df_context.columns:
            context_attempt_weight = float(row_weights(df_context).sum())
            ctx_error_weight = 0.0
            if "isWide" in df_context.columns:
                ctx_error_weight += float(row_weights(df_context[df_context["isWide"].isin(["true", "1"])]).sum())
            if "isNoBall" in df_context.columns:
                ctx_error_weight += float(row_weights(df_context[df_context["isNoBall"].isin(["true", "1"])]).sum())
            context_error_rate = ctx_error_weight / context_attempt_weight if context_attempt_weight else 0.0

        sm_dot = smooth_rate(weighted_dots, effective_balls, prior_dot_rate, BOWLER_PRIOR_BALLS)
        sm_bdy = smooth_rate(weighted_boundaries, effective_balls, prior_boundary_rate, BOWLER_PRIOR_BALLS)
        sm_wkt = smooth_rate(weighted_wickets, effective_balls, prior_wicket_rate, BOWLER_PRIOR_BALLS)
        sm_contain = smooth_rate(weighted_contain, effective_balls, prior_contain_rate, BOWLER_PRIOR_BALLS)
        sm_rpb = smooth_runs(weighted_runs, effective_balls, prior_runs_per_ball, BOWLER_PRIOR_BALLS)
        sm_error = smooth_rate(
            weighted_err_count, weighted_total_attempts, context_error_rate, BOWLER_PRIOR_BALLS
        )

        row = {c_name: str(k) for c_name, k in zip(legal_cols, keys)}
        row.update({
            "bowler_exec_balls": b,
            "bowler_exec_effective_balls": round(effective_balls, 1),
            "bowler_exec_runs": r,
            "bowler_exec_wickets": int(g.apply(is_out, axis=1).sum()),
            "bowler_exec_error_pct": pct(err_count, total_attempts),
            "bowler_smoothed_dot_pct": pct(sm_dot, 1),
            "bowler_smoothed_boundary_pct": pct(sm_bdy, 1),
            "bowler_smoothed_wkt_pct": pct(sm_wkt, 1),
            "bowler_smoothed_contain_pct": pct(sm_contain, 1),
            "bowler_smoothed_error_pct": pct(sm_error, 1),
            "bowler_smoothed_runs_per_ball": round(sm_rpb, 3),
        })
        records.append(row)

    return pd.DataFrame(records) if records else pd.DataFrame()


def add_bowler_execution(dtable: pd.DataFrame,
                         bowler_exec: pd.DataFrame) -> pd.DataFrame:
    if dtable.empty or bowler_exec.empty:
        return dtable
    merge_cols = [c for c in DELIVERY_COLS if c in dtable.columns and c in bowler_exec.columns]
    if not merge_cols:
        return dtable
    return dtable.merge(bowler_exec, on=merge_cols, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic intent engine
# ─────────────────────────────────────────────────────────────────────────────
def ground_phase_avg_rr(df_all: pd.DataFrame, ground: str, lo: int, hi: int) -> float | None:
    """
    Average run rate (runs per over) at the given ground for overs lo–hi,
    across the last 10 matches. Returns None if data is insufficient.
    """
    if not ground or "ground" not in df_all.columns or "fixtureId" not in df_all.columns:
        return None
    df_g = df_all[
        (df_all["ground"] == ground) &
        (df_all["overNumber"] >= lo) &
        (df_all["overNumber"] <= hi)
    ].copy()
    if df_g.empty:
        return None
    last_10 = df_g["fixtureId"].drop_duplicates().sort_values().tail(10)
    df_g = df_g[df_g["fixtureId"].isin(last_10)]
    df_g = df_g[~df_g.get("isWide", pd.Series(["false"] * len(df_g))).isin(["true", "1"])]
    weights = row_weights(df_g)
    effective_balls = float(weights.sum())
    if len(df_g) < 10 or effective_balls <= 0:
        return None
    return round(float((df_g["batruns"] * weights).sum()) / effective_balls * 6, 2)


def determine_intent(
    over: int, innings: str, wickets_down: int,
    runs_needed: int, balls_left: int,
    batter_sr: float,
    team_crr: float = 0.0,
    ground_phase_rr: float | None = None,
) -> tuple[str, str]:
    """
    Returns (intent, reason) based purely on match situation.
    intent is one of: "WICKET", "CONTAIN"
    """
    phase    = phase_of(over)
    is_chase = "chasing" in innings.lower()
    rrr      = round(runs_needed / balls_left * 6, 2) if balls_left else 0
    crr      = round(team_crr, 2)

    if is_chase and phase == "Death":
        return "CONTAIN", f"Chasing · Death · chasing → restrict total"
    
    elif is_chase and phase == "Powerplay" and rrr>crr:
        return "WICKET", f"Chasing · Powerplay · RRR {rrr} > team CRR {crr} → Better chance of wicket + early pressure"
    
    elif is_chase:
        return "CONTAIN", f"Chasing · {phase} · team CRR {crr} ≥ RRR {rrr} or outside powerplay wicket trigger → restrict total"

    elif not is_chase and phase == "Powerplay":
        if ground_phase_rr is not None and team_crr > ground_phase_rr:
            return "CONTAIN", (
                f"Powerplay · team Current Run Rate {team_crr} > ground avg {ground_phase_rr} rpo "
                f"→ scoring above par → limiting runs priority"
            )
        elif ground_phase_rr is not None and team_crr <= ground_phase_rr:
            return "WICKET", (
                f"Powerplay · team Current Run Rate {team_crr} ≤ ground avg {ground_phase_rr} rpo "
                f"→ scoring at/under par → wicket-taking opportunity"
            )
        else:
            return "CONTAIN", "Powerplay · restrict scoring platform → dot + control"

    else:
        return "CONTAIN", "Default Strategy for non-chase middle/death overs → build pressure and wait for mistakes"


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic scoring function
# ─────────────────────────────────────────────────────────────────────────────
def empirical_score_delivery(row: dict, intent: str) -> float:
    """
    Scores a composite delivery fingerprint row for a given intent.

    Fairness mechanics:
    - Bayesian smoothing: small delivery samples are pulled toward the current
      batter/context baseline before ranking.
    - Smooth confidence: balls/MIN_RELIABLE, capped at 1.0. Monotonic in
      sample size, no cliff at the reliability boundary.
    - Wicket bonus is confidence-weighted so tiny wicket samples do not dominate.
    """
    balls        = float(row.get("effective_balls", row["balls"]))
    confidence   = min(1.0, balls / MIN_RELIABLE)
    wicket_count = float(row.get("weighted_wickets", row["wickets"]))
    wicket_bonus = min(wicket_count * 5, 20) * confidence
    wkt_pct = row.get("smoothed_wkt_pct", row["wkt_pct"])
    dot_pct = row.get("smoothed_dot_pct", row["dot_pct"])
    contained_pct = row.get("smoothed_contained_ball_pct", row["contained_ball_pct"])
    sr_value = row.get("smoothed_sr", row["sr"])

    if intent == "WICKET":
        raw = (wkt_pct * 0.60
               + dot_pct * 0.20
               + max(0, 100 - sr_value) * 0.20
               + wicket_bonus)
    else:  # CONTAIN
        raw = (contained_pct * 0.80
               + (100 / (1 + sr_value / 100)) * 0.20
               + wicket_bonus * 0.5)  # wicket bonus counts half for CONTAIN

    return raw * confidence


def _has_bowler_execution(row: dict) -> bool:
    needed = [
        "bowler_smoothed_dot_pct", "bowler_smoothed_boundary_pct",
        "bowler_smoothed_wkt_pct", "bowler_smoothed_contain_pct",
        "bowler_smoothed_error_pct", "bowler_smoothed_runs_per_ball",
    ]
    return all(k in row and pd.notna(row[k]) for k in needed)


def score_bowler_execution(row: dict, intent: str) -> float:
    dot_pct = float(row["bowler_smoothed_dot_pct"])
    bdy_pct = float(row["bowler_smoothed_boundary_pct"])
    wkt_pct = float(row["bowler_smoothed_wkt_pct"])
    contain_pct = float(row["bowler_smoothed_contain_pct"])
    err_pct = float(row["bowler_smoothed_error_pct"])
    rpb = float(row["bowler_smoothed_runs_per_ball"])

    if intent == "WICKET":
        score = (
            wkt_pct * 0.45
            + dot_pct * 0.25
            + max(0, 100 - bdy_pct) * 0.15
            + max(0, 100 - err_pct) * 0.10
            + max(0, 2 - rpb) * 5
        )
    else:
        score = (
            contain_pct * 0.45
            + dot_pct * 0.25
            + max(0, 100 - bdy_pct) * 0.15
            + max(0, 100 - err_pct) * 0.10
            + max(0, 2 - rpb) * 5
        )
    return max(0.0, score)


def score_delivery(row: dict, intent: str) -> float:
    """
    Main delivery score.
    Uses empirical batter evidence, then blends in selected bowler execution
    reliability when available.
    """
    empirical = empirical_score_delivery(row, intent)
    if _has_bowler_execution(row):
        bowler_balls = row.get("bowler_exec_effective_balls", row.get("bowler_exec_balls", 0))
        bowler_confidence = min(1.0, float(bowler_balls) / MIN_RELIABLE)
        bowler_weight = BOWLER_EXECUTION_WEIGHT * bowler_confidence
        return bowler_weight * score_bowler_execution(row, intent) + (1 - bowler_weight) * empirical
    return empirical


def score_delivery_breakdown(row: dict, intent: str) -> dict:
    """
    Return the same score calculation as score_delivery(), split into readable
    pieces for UI explanation.
    """
    balls = float(row.get("effective_balls", row.get("balls", 0)))
    confidence = min(1.0, balls / MIN_RELIABLE)
    wicket_count = float(row.get("weighted_wickets", row.get("wickets", 0)))
    wicket_bonus = min(wicket_count * 5, 20) * confidence
    wkt_pct = row.get("smoothed_wkt_pct", row["wkt_pct"])
    dot_pct = row.get("smoothed_dot_pct", row["dot_pct"])
    contained_pct = row.get("smoothed_contained_ball_pct", row["contained_ball_pct"])
    sr_value = row.get("smoothed_sr", row["sr"])

    if intent == "WICKET":
        components = {
            "wicket_component": wkt_pct * 0.60,
            "dot_component": dot_pct * 0.20,
            "low_sr_component": max(0, 100 - sr_value) * 0.20,
            "wicket_bonus": wicket_bonus,
        }
    else:
        components = {
            "contain_component": contained_pct * 0.80,
            "low_sr_component": (100 / (1 + sr_value / 100)) * 0.20,
            "wicket_bonus": wicket_bonus * 0.5,
        }

    raw_score = sum(components.values())
    empirical = raw_score * confidence
    final_score = empirical
    bowler_weight = 0.0
    bowler_score = None

    if _has_bowler_execution(row):
        bowler_balls = row.get("bowler_exec_effective_balls", row.get("bowler_exec_balls", 0))
        bowler_confidence = min(1.0, float(bowler_balls) / MIN_RELIABLE)
        bowler_weight = BOWLER_EXECUTION_WEIGHT * bowler_confidence
        bowler_score = score_bowler_execution(row, intent)
        final_score = bowler_weight * bowler_score + (1 - bowler_weight) * empirical

    return {
        **components,
        "confidence": confidence,
        "raw_score": raw_score,
        "empirical_score": empirical,
        "bowler_weight": bowler_weight,
        "bowler_score": bowler_score,
        "final_score": final_score,
    }


def _num(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fingerprint_key(row, cols: list[str]) -> tuple:
    return tuple(str(row.get(c, "")) for c in cols)


def apply_partial_matchup_adjustment(
    dtable_base: pd.DataFrame,
    dtable_matchup: pd.DataFrame,
    matchup_total_balls: int,
    intent: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Keep broader batter-vs-bowler-type data as the base table, but let small
    direct bowler-vs-batter evidence make a capped adjustment when it is
    specific and meaningfully different.
    """
    summary = {"applied": 0, "rows": []}
    if (
        dtable_base.empty
        or dtable_matchup.empty
        or matchup_total_balls < PARTIAL_MATCHUP_MIN_TOTAL_BALLS
    ):
        return dtable_base, summary

    base = dtable_base.copy()
    available_cols = [
        c for c in DELIVERY_COLS
        if c in base.columns and c in dtable_matchup.columns
    ]
    if not available_cols:
        return base, summary

    base["partial_matchup_used"] = False
    base["partial_matchup_weight"] = None
    base["partial_matchup_balls"] = None
    base["partial_matchup_effective_balls"] = None
    base["partial_matchup_wickets"] = None
    base["partial_matchup_score_delta"] = None
    base["partial_matchup_direction"] = ""
    base["partial_matchup_reason"] = ""

    matchup_lookup = {
        _fingerprint_key(row, available_cols): row
        for _, row in dtable_matchup.iterrows()
    }

    for idx, row in base.iterrows():
        key = _fingerprint_key(row, available_cols)
        mu = matchup_lookup.get(key)
        if mu is None:
            continue

        mu_balls = int(_num(mu.get("balls", 0)))
        mu_wickets = int(_num(mu.get("wickets", 0)))
        if (
            mu_balls < PARTIAL_MATCHUP_MIN_DELIVERY_BALLS
            and mu_wickets < PARTIAL_MATCHUP_MIN_DELIVERY_WICKETS
        ):
            continue

        base_sr = _num(row.get("smoothed_sr", row.get("sr", 0)))
        mu_sr = _num(mu.get("smoothed_sr", mu.get("sr", 0)))
        base_dot = _num(row.get("smoothed_dot_pct", row.get("dot_pct", 0)))
        mu_dot = _num(mu.get("smoothed_dot_pct", mu.get("dot_pct", 0)))
        base_bdy = _num(row.get("smoothed_boundary_pct", row.get("boundary_pct", 0)))
        mu_bdy = _num(mu.get("smoothed_boundary_pct", mu.get("boundary_pct", 0)))

        reasons = []
        if abs(mu_sr - base_sr) >= PARTIAL_MATCHUP_SR_DIFF:
            direction = "higher" if mu_sr > base_sr else "lower"
            reasons.append(f"direct SR {mu_sr} is {direction} than base {base_sr}")
        if abs(mu_dot - base_dot) >= PARTIAL_MATCHUP_DOT_DIFF:
            direction = "higher" if mu_dot > base_dot else "lower"
            reasons.append(f"direct dot% {mu_dot} is {direction} than base {base_dot}")
        if abs(mu_bdy - base_bdy) >= PARTIAL_MATCHUP_BOUNDARY_DIFF:
            direction = "higher" if mu_bdy > base_bdy else "lower"
            reasons.append(f"direct boundary% {mu_bdy} is {direction} than base {base_bdy}")
        if mu_wickets >= PARTIAL_MATCHUP_MIN_DELIVERY_WICKETS:
            reasons.append(f"{mu_wickets} direct wicket(s)")

        if not reasons:
            continue

        mu_effective = max(0.0, _num(mu.get("effective_balls", mu_balls)))
        weight = min(
            PARTIAL_MATCHUP_MAX_WEIGHT,
            mu_effective / PARTIAL_MATCHUP_CONFIDENCE_BALLS,
        )
        if weight <= 0:
            continue

        before_score = empirical_score_delivery(row.to_dict(), intent)

        for col in [
            "smoothed_wkt_pct",
            "smoothed_dot_pct",
            "smoothed_boundary_pct",
            "smoothed_contained_ball_pct",
        ]:
            if col in base.columns and col in dtable_matchup.columns:
                base_val = _num(row.get(col))
                mu_val = _num(mu.get(col))
                base.at[idx, f"base_{col}"] = base_val
                base.at[idx, col] = round((1 - weight) * base_val + weight * mu_val, 1)

        base_rpb = _num(row.get("smoothed_runs_per_ball", row.get("sr", 0) / 100))
        mu_rpb = _num(mu.get("smoothed_runs_per_ball", mu.get("sr", 0) / 100))
        new_rpb = (1 - weight) * base_rpb + weight * mu_rpb
        base.at[idx, "base_smoothed_runs_per_ball"] = base_rpb
        base.at[idx, "smoothed_runs_per_ball"] = round(new_rpb, 3)
        base.at[idx, "smoothed_sr"] = round(new_rpb * 100, 1)

        after_score = empirical_score_delivery(base.loc[idx].to_dict(), intent)
        score_delta = round(after_score - before_score, 2)
        direction = "boosted" if score_delta > 0 else "reduced"

        base.at[idx, "partial_matchup_used"] = True
        base.at[idx, "partial_matchup_weight"] = round(weight, 3)
        base.at[idx, "partial_matchup_balls"] = mu_balls
        base.at[idx, "partial_matchup_effective_balls"] = round(mu_effective, 1)
        base.at[idx, "partial_matchup_wickets"] = mu_wickets
        base.at[idx, "partial_matchup_score_delta"] = score_delta
        base.at[idx, "partial_matchup_direction"] = direction
        base.at[idx, "partial_matchup_reason"] = "; ".join(reasons[:2])

        summary["applied"] += 1
        summary["rows"].append({
            "delivery": " · ".join(str(row.get(c, "—")) for c in available_cols),
            "direction": direction,
            "score_delta": score_delta,
            "balls": mu_balls,
            "weight": round(weight, 3),
        })

    return base, summary


# ─────────────────────────────────────────────────────────────────────────────
# Secondary variation lookup — within a fixed line+length
# ─────────────────────────────────────────────────────────────────────────────
def recommend_variation(df: pd.DataFrame, length: str, line: str,
                        intent: str, n: int = 2) -> list:
    """
    Given a chosen line+length, find the best bowlingDetailId variation
    within that combination.

    Uses recency-weighted evidence so older variation outcomes do not count
    equally with recent outcomes.
    """
    if "bowlingDetailId" not in df.columns:
        return []

    mask = (
        (df["lengthTypeId"] == length) &
        (df["lineTypeId"] == line) &
        (~df.get("isWide", pd.Series(["false"] * len(df))).isin(["true", "1"]))
    )
    sub = df[mask].copy()
    if sub.empty:
        return []

    sub_weights = row_weights(sub)
    prior_balls = float(sub_weights.sum())
    if prior_balls <= 0:
        return []

    prior_runs = float((sub["batruns"] * sub_weights).sum())
    prior_wickets = float(sub_weights[sub.apply(is_out, axis=1)].sum())
    prior_dots = float(sub_weights[sub["batruns"] == 0].sum())
    prior_contain = float(sub_weights[sub["batruns"].isin([0, 1, 2])].sum())

    prior_runs_per_ball = prior_runs / prior_balls
    prior_wicket_rate = prior_wickets / prior_balls
    prior_dot_rate = prior_dots / prior_balls
    prior_contain_rate = prior_contain / prior_balls

    results = []
    for var, g in sub.groupby("bowlingDetailId"):
        var = str(var).strip()
        if var.lower() in ("unknown", "nan", "none", ""):
            continue

        b = len(g)
        gw = row_weights(g)
        effective_balls = float(gw.sum())
        if effective_balls <= 0:
            continue

        raw_runs = int(g["batruns"].sum())
        raw_wickets = int(g.apply(is_out, axis=1).sum())

        weighted_runs = float((g["batruns"] * gw).sum())
        weighted_wickets = float(gw[g.apply(is_out, axis=1)].sum())
        weighted_dots = float(gw[g["batruns"] == 0].sum())
        weighted_contain = float(gw[g["batruns"].isin([0, 1, 2])].sum())

        sm_rpb = smooth_runs(
            weighted_runs, effective_balls, prior_runs_per_ball, RUNS_PRIOR_BALLS
        )
        sm_wkt = smooth_rate(
            weighted_wickets, effective_balls, prior_wicket_rate, WICKET_PRIOR_BALLS
        )
        sm_dot = smooth_rate(
            weighted_dots, effective_balls, prior_dot_rate, DOT_PRIOR_BALLS
        )
        sm_contain = smooth_rate(
            weighted_contain, effective_balls, prior_contain_rate, CONTAIN_PRIOR_BALLS
        )

        stats = {
            "balls": b,
            "effective_balls": round(effective_balls, 1),
            "runs": raw_runs,
            "sr": sr(weighted_runs, effective_balls),
            "wkt_pct": pct(weighted_wickets, effective_balls),
            "dot_pct": pct(weighted_dots, effective_balls),
            "contained_ball_pct": pct(weighted_contain, effective_balls),
            "smoothed_sr": round(sm_rpb * 100, 1),
            "smoothed_wkt_pct": pct(sm_wkt, 1),
            "smoothed_dot_pct": pct(sm_dot, 1),
            "smoothed_contained_ball_pct": pct(sm_contain, 1),
            "wickets": raw_wickets,
        }

        confidence = min(1.0, effective_balls / MIN_RELIABLE)
        if intent == "WICKET":
            raw = (
                stats["smoothed_wkt_pct"] * 0.60
                + stats["smoothed_dot_pct"] * 0.20
                + max(0, 100 - stats["smoothed_sr"]) * 0.20
            )
        else:  # CONTAIN
            raw = (
                stats["smoothed_contained_ball_pct"] * 0.80
                + max(0, 100 - stats["smoothed_sr"]) * 0.20
            )

        score = raw * confidence
        results.append((var, stats, score))

    results.sort(key=lambda x: -x[2])
    return [(var, stats) for var, stats, _ in results[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic 6-ball plan builder
# ─────────────────────────────────────────────────────────────────────────────
def build_plan(
    dtable: pd.DataFrame,
    intent: str,
    intent_reason: str,
    sequences: list | None = None,
    df_raw: pd.DataFrame | None = None,
    phase: str | None = None,
) -> dict:
    """
    Builds the 6-ball plan entirely from the delivery table.
    No LLM involved — pure deterministic selection.

    Selection logic:
    - Greedy with adaptation cap: each ball picks the highest-scored
      non-danger fingerprint that hasn't been used MAX_REPEATS times yet.
      Models the reality that batters calibrate to repeated deliveries.
    - Phase-aware sequence override: if a phase-matched setup→wicket pair
      exists, place it where it makes tactical sense:
        Powerplay → balls 1-2 (attack while batter is fresh)
        Middle    → balls 5-6 (build pressure, then attack)
        Death     → disabled (execution-driven, not setup-driven)
    - Danger zone (highest-SR fingerprint) is always excluded.
    """
    if dtable.empty:
        return {"error": "Insufficient data to build a plan."}

    rows = dtable.to_dict("records")
    rows = [
        r for r in rows
        if int(r.get("balls", 0)) >= MIN_RELIABLE or int(r.get("wickets", 0)) >= 2
    ]
    if not rows:
        return {
            "error": (
                "Insufficient data: no composite delivery has enough raw evidence "
                f"({MIN_RELIABLE}+ balls or 2+ wickets)."
            )
        }

    available_cols = [c for c in DELIVERY_COLS if c in dtable.columns]

    # Score every composite fingerprint
    for row in rows:
        row["_score"] = score_delivery(row, intent)

    reliable   = [r for r in rows if r["reliable"]]
    unreliable = [r for r in rows if not r["reliable"]]
    pool = reliable if reliable else unreliable
    pool_sorted = sorted(pool, key=lambda r: -r["_score"])

    # Danger zone: delivery where batter scores most freely.
    # Prefer reliable effective samples and use smoothed SR so tiny raw samples
    # do not create exaggerated "never bowl" warnings.
    danger_pool = [
        r for r in rows
        if float(r.get("effective_balls", r["balls"])) >= MIN_RELIABLE
    ]
    danger_candidates = danger_pool if danger_pool else rows
    danger = max(
        danger_candidates,
        key=lambda r: (
            float(r.get("smoothed_sr", r["sr"])),
            float(r.get("boundary_pct", 0)),
            float(r.get("effective_balls", r["balls"])),
        ),
    )

    def fp_key(r):
        return (r.get("lengthTypeId", ""), r.get("lineTypeId", ""),
                r.get("bowlingFromId", ""))

    def is_danger(r):
        return (r.get("lengthTypeId") == danger.get("lengthTypeId")
                and r.get("lineTypeId") == danger.get("lineTypeId")
                and r.get("bowlingFromId") == danger.get("bowlingFromId"))

    # ── Top non-danger pick (greedy default for every ball) ─────────────────
    top_pick = None
    for r in pool_sorted:
        if not is_danger(r):
            top_pick = r
            break
    if top_pick is None:
        # All candidates are danger zone — fall back to the highest-scored
        top_pick = pool_sorted[0]

    # ── Mine sequence patterns ───────────────────────────────────────────────
    sequence_setup = None
    sequence_wicket = None
    sequence_meta = None

    if sequences:
        # Filter sequences by phase if provided
        seqs_to_use = sequences
        if phase:
            seqs_to_use = [s for s in sequences if s.get("phase") == phase]

        for seq in seqs_to_use:
            wk_fp = _fp_tuple(seq["wicket"], available_cols)

            matched_wicket = None
            for r in pool:
                if _fp_tuple(r, available_cols) == wk_fp and not is_danger(r):
                    matched_wicket = r
                    break
            if matched_wicket is None:
                continue

            matched_setup = None
            # Prefer the immediate setup ball from a 3-ball pattern; otherwise
            # use the single setup ball from a 2-ball pattern.
            for setup_item in reversed(seq["setup"]):
                setup_fp = _fp_tuple(setup_item, available_cols)
                for r in pool:
                    if (_fp_tuple(r, available_cols) == setup_fp
                            and not is_danger(r)
                            and r is not matched_wicket):
                        matched_setup = r
                        break
                if matched_setup:
                    break

            if matched_setup is not None:
                sequence_setup = matched_setup
                sequence_wicket = matched_wicket
                sequence_meta = seq
                break

    # Sequences only useful as a complete pair: setup → wicket
    has_pair = sequence_setup is not None and sequence_wicket is not None

    # Intent-aware sequence placement:
    # - WICKET: use the setup pair early to attack immediately.
    # - CONTAIN: use the setup pair late after pressure has been built.
    # - Death: disable sequences unless the current intent is explicitly WICKET,
    #   because death-over plans are usually execution/risk-control driven.
    setup_slot: int | None = None
    wicket_slot: int | None = None
    if has_pair:
        if phase == "Death" and intent != "WICKET":
            has_pair = False
        elif intent == "WICKET":
            setup_slot, wicket_slot = 1, 2
        else:
            setup_slot, wicket_slot = 5, 6
    has_sequence = has_pair

    # ── Variation lookup cache (avoids redundant calls when same pick repeats) ─
    var_cache: dict = {}

    def get_variations(length: str, line: str):
        cache_key = (length, line)
        if cache_key not in var_cache:
            if df_raw is not None and not df_raw.empty:
                var_cache[cache_key] = recommend_variation(df_raw, length, line, intent, n=2)
            else:
                var_cache[cache_key] = []
        return var_cache[cache_key]

    # ── Build 6 balls: greedy best, with sequence override at balls 5-6 ─────
    intent_tag = "Wicket ball" if intent == "WICKET" else "Contain"
    usage_counts: dict = {}

    balls = []
    for ball_num in range(1, 7):
        # Sequence override: phase-determined slot positions (powerplay: 1-2, middle: 5-6, death: off)
        if has_pair and ball_num == setup_slot:
            pick = sequence_setup
            role = "setup"
            tag = "Set up"
        elif has_pair and ball_num == wicket_slot:
            pick = sequence_wicket
            role = "wicket"
            tag = "Wicket ball"
        else:
            # Greedy with MAX_REPEATS cap: pick best non-danger that hasn't hit the cap
            pick = None
            for r in pool_sorted:
                if is_danger(r):
                    continue
                if usage_counts.get(fp_key(r), 0) >= MAX_REPEATS:
                    continue
                pick = r
                break
            if pick is None:
                pick = top_pick  # all candidates capped — fall back
            role = "best"
            tag = intent_tag

        # Track usage (used by the cap above; also surfaced for debugging)
        key = fp_key(pick)
        usage_counts[key] = usage_counts.get(key, 0) + 1

        # Secondary variation lookup within the chosen line+length (cached)
        var_recs = get_variations(pick.get("lengthTypeId", ""), pick.get("lineTypeId", ""))
        top_var       = var_recs[0][0] if var_recs else "—"
        top_var_stats = var_recs[0][1] if var_recs else {}
        alt_vars      = var_recs[1:]

        balls.append({
            "ball":        ball_num,
            "length":      pick.get("lengthTypeId", "—"),
            "line":        pick.get("lineTypeId",   "—"),
            "variation":   top_var,
            "variation_stats": top_var_stats,
            "variation_alts":  [(v, s) for v, s in alt_vars],
            "from_wicket": pick.get("bowlingFromId",   "—"),
            "hand":        pick.get("bowlingHandId",   "—"),
            "fields":      pick["top_fields"],
            "intent":      tag,
            "role":        role,
            "stats": {
                "balls":        pick["balls"],
                "effective_balls": pick.get("effective_balls"),
                "sr":           pick["sr"],
                "wkt_pct":      pick["wkt_pct"],
                "dot_pct":      pick["dot_pct"],
                "boundary_pct": pick["boundary_pct"],
                "smoothed_wkt_pct": pick.get("smoothed_wkt_pct"),
                "smoothed_dot_pct": pick.get("smoothed_dot_pct"),
                "smoothed_boundary_pct": pick.get("smoothed_boundary_pct"),
                "smoothed_contained_ball_pct": pick.get("smoothed_contained_ball_pct"),
                "smoothed_sr": pick.get("smoothed_sr"),
                "wickets":      pick["wickets"],
                "reliable":     pick["reliable"],
                "bowler_exec_balls": pick.get("bowler_exec_balls"),
                "bowler_exec_effective_balls": pick.get("bowler_exec_effective_balls"),
                "bowler_smoothed_dot_pct": pick.get("bowler_smoothed_dot_pct"),
                "bowler_smoothed_boundary_pct": pick.get("bowler_smoothed_boundary_pct"),
                "bowler_smoothed_wkt_pct": pick.get("bowler_smoothed_wkt_pct"),
                "bowler_smoothed_contain_pct": pick.get("bowler_smoothed_contain_pct"),
                "bowler_smoothed_error_pct": pick.get("bowler_smoothed_error_pct"),
                "bowler_smoothed_runs_per_ball": pick.get("bowler_smoothed_runs_per_ball"),
                "partial_matchup_used": pick.get("partial_matchup_used", False),
                "partial_matchup_weight": pick.get("partial_matchup_weight"),
                "partial_matchup_balls": pick.get("partial_matchup_balls"),
                "partial_matchup_effective_balls": pick.get("partial_matchup_effective_balls"),
                "partial_matchup_wickets": pick.get("partial_matchup_wickets"),
                "partial_matchup_score_delta": pick.get("partial_matchup_score_delta"),
                "partial_matchup_direction": pick.get("partial_matchup_direction"),
                "partial_matchup_reason": pick.get("partial_matchup_reason"),
            },
            "score": round(pick["_score"], 2),
            "why":   "",
        })

    danger_count = sum(
        1 for b in balls
        if b.get("length") == danger.get("lengthTypeId")
        and b.get("line") == danger.get("lineTypeId")
        and b.get("from_wicket") == danger.get("bowlingFromId")
    )

    return {
        "intent":               intent,
        "intent_reason":        intent_reason,
        "balls":                balls,
        "danger": {
            "length": danger.get("lengthTypeId", "—"),
            "line":   danger.get("lineTypeId",   "—"),
            "from":   danger.get("bowlingFromId",   "—"),
            "hand":   danger.get("bowlingHandId",   "—"),
            "sr":     danger["sr"],
            "smoothed_sr": danger.get("smoothed_sr", danger["sr"]),
            "balls":  danger["balls"],
            "effective_balls": danger.get("effective_balls", danger["balls"]),
            "reliable": danger.get("reliable", False),
        },
        "danger_used":          danger_count > 0,
        "danger_count":         danger_count,
        "used_unreliable":      not bool(reliable),
        "has_sequence_pattern":  has_sequence,
        "sequence_meta":         sequence_meta if has_sequence else None,
        "phase":                phase,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM narration — one sentence per ball, decisions are locked
# ─────────────────────────────────────────────────────────────────────────────
def narrate_plan(plan: dict, batter: str, bowler: str,
                 bowler_type: str, bowler_hand: str,
                 context: str) -> dict:
    """
    Sends pre-decided ball data to Groq.
    LLM can ONLY write the 'why' sentence. All delivery choices are final.
    """
    ball_briefs = []
    for b in plan["balls"]:
        s = b["stats"]
        var_stats = b.get("variation_stats", {})
        var_note = (f"{b['variation']} (Wkt% {var_stats.get('wkt_pct','?')}, "
                    f"Dot% {var_stats.get('dot_pct','?')}, N={var_stats.get('balls','?')})"
                    if b["variation"] not in ("—", "") else "No variation data")
        alt_vars = b.get("variation_alts", [])

        ball_briefs.append({
            "ball":               b["ball"],
            "line_length":        f"{b['length']} · {b['line']} · {b['from_wicket']} · {b['hand']}",
            "recommended_variation": var_note,
            "alt_variation":      f"{alt_vars[0][0]} (Wkt% {alt_vars[0][1].get('wkt_pct','?')}, N={alt_vars[0][1].get('balls','?')})" if alt_vars else "—",
            "fields":             b["fields"],
            "intent":             b["intent"],
            "role":               b.get("role", "best"),
            "line_length_stats":  s,
            "reliable":           s["reliable"],
        })

    danger = plan["danger"]
    has_seq = plan.get("has_sequence_pattern", False)

    def _sanitize(s: str) -> str:
        """Replace non-ASCII symbols that cause LLM JSON output to break."""
        return (
            str(s)
            .replace("≤", "<=")
            .replace("≥", ">=")
            .replace("→", "->")
            .replace("·", "-")
        )

    safe_intent_reason = _sanitize(plan["intent_reason"])

    # Sanitize all string fields in ball_briefs
    for brief in ball_briefs:
        for k, v in brief.items():
            if isinstance(v, str):
                brief[k] = _sanitize(v)

    prompt = f"""You are a cricket data analyst writing a match-day bowling brief.

The delivery decisions below have already been made by a deterministic algorithm.
Your ONLY job is to write ONE plain-English sentence per ball explaining WHY
that delivery was chosen — using ONLY the numbers provided.

STRICT RULES:
- Do not change, question, or add to any delivery decision.
- Do not use any cricket knowledge not reflected in the numbers.
- If reliable=false, the sentence MUST start with: "(Small sample, N=X) ..."
- Keep each sentence under 35 words.
- Write in second person: "Bowl a ..."
- Each ball has TWO layers of decision:
    1. line_length — the primary choice (large sample, high confidence)
    2. recommended_variation — the best variation WITHIN that line+length (secondary lookup)
  Your sentence should mention BOTH: the line+length and the recommended variation.
- Each ball has a ROLE that explains its tactical purpose:
  - "dot" — pressure building, emphasise dot% and low SR
  - "wicket" — attacking delivery, emphasise wkt% and dismissal data
  - "setup" — creates a pattern the batter expects, mention it sets up the next ball
  - "surprise" — breaks the pattern established by setup balls, mention the change of angle/line/length
  - "best" — highest-scoring option for the current intent
{"- This plan uses SEQUENCE PATTERNS mined from historical wicket-taking data. Mention this for setup/surprise balls." if has_seq else ""}

CONTEXT: {context}
BATTER: {batter}
BOWLER: {bowler if bowler else "Not specified"} - {bowler_hand}-arm {bowler_type}
INTENT THIS OVER: {plan["intent"]} - {safe_intent_reason}

BALLS TO NARRATE:
{json.dumps(ball_briefs, indent=2)}

DANGER ZONE (never bowl this line+length — highest SR in data):
{danger["length"]} - {danger["line"]} - from {danger["from"]} - {danger["hand"]}
Raw SR: {danger["sr"]}, sample: {danger["balls"]} balls

Return ONLY valid JSON — no markdown, no text outside JSON:
{{
  "ball_explanations": [
    {{"ball": 1, "why": "..."}},
    {{"ball": 2, "why": "..."}},
    {{"ball": 3, "why": "..."}},
    {{"ball": 4, "why": "..."}},
    {{"ball": 5, "why": "..."}},
    {{"ball": 6, "why": "..."}}
  ],
  "danger_explanation": "One sentence: why to avoid the danger zone delivery, citing SR and sample size.",
  "overall_note": "One sentence summary of what the data shows about this batter in this situation."
}}"""

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    def _parse(s: str) -> dict:
        # Strip trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", s)
        # Try standard parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try extracting outermost JSON object
        m = re.search(r'\{[\s\S]*\}', cleaned)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Fallback: extract ball explanations via regex
        result: dict = {"ball_explanations": []}
        for bm in re.finditer(r'"ball"\s*:\s*(\d+)\s*,\s*"why"\s*:\s*"([^"]*)"', cleaned):
            result["ball_explanations"].append({"ball": int(bm.group(1)), "why": bm.group(2)})
        dn = re.search(r'"danger_explanation"\s*:\s*"([^"]*)"', cleaned)
        if dn:
            result["danger_explanation"] = dn.group(1)
        on = re.search(r'"overall_note"\s*:\s*"([^"]*)"', cleaned)
        if on:
            result["overall_note"] = on.group(1)
        return result

    candidates = [raw]
    if "```" in raw:
        candidates = [p.strip().lstrip("json").strip() for p in raw.split("```")]
    for part in candidates:
        try:
            return _parse(part)
        except Exception:
            continue
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Stats summary for data preview
# ─────────────────────────────────────────────────────────────────────────────
def summary_stats(df: pd.DataFrame) -> dict:
    df = df[~df.get("isWide", pd.Series(["false"]*len(df))).isin(["true", "1"])].copy()
    b   = len(df)
    weights = row_weights(df)
    effective_balls = float(weights.sum())
    r   = float((df["batruns"] * weights).sum())
    wicket_mask = df.apply(is_out, axis=1)
    w   = int(wicket_mask.sum())
    weighted_w = float(weights[wicket_mask].sum())
    bdy = float(weights[df["batruns"].isin([4, 6])].sum())
    d   = float(weights[df["batruns"] == 0].sum())
    return {
        "balls": b, "effective_balls": round(effective_balls, 1),
        "runs": round(r, 1), "sr": sr(r, effective_balls),
        "wickets": w, "weighted_wickets": round(weighted_w, 1),
        "boundary_pct": pct(bdy, effective_balls),
        "dot_pct": pct(d, effective_balls),
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_chips(s: dict):
    st.markdown(f"""
    <div class="stat-row">
      <div class="chip"><span>SR</span>{s['sr']}</div>
      <div class="chip"><span>Balls</span>{s['balls']}</div>
      <div class="chip"><span>Bdry%</span>{s['boundary_pct']}%</div>
      <div class="chip"><span>Dot%</span>{s['dot_pct']}%</div>
      <div class="chip"><span>Wkts</span>{s['wickets']}</div>
    </div>""", unsafe_allow_html=True)


def render_app_methodology():
    with st.expander("How this app works", expanded=False):
        st.markdown("""
        1. The app filters historical balls by batter, bowler type, bowler arm, phase, and ground.
        2. If a bowler is selected, direct matchup data is checked separately.
           Strong direct matchup data can replace the base table. Sparse but
           meaningful direct matchup data can only make a capped partial
           adjustment.
        3. If the batter has thin data, similar batters are added using delivery-vulnerability profiles.
        4. Older seasons are down-weighted using recency weighting.
        5. Small samples are smoothed toward wider context averages.
        6. Each delivery type is scored for the current intent: CONTAIN or WICKET.
        7. If selected bowler data exists, bowler execution adjusts the score.
        8. The plan avoids the danger-zone delivery and may use wicket-sequence patterns.
        """)


def render_ds_methodology():
    with st.expander("Data Science methods used in this app", expanded=False):
        st.markdown(f"""
        **1. Similar batter model**

        Used when the selected batter has fewer than `{MIN_SIMILAR_THRESH}` balls
        in the chosen context. The app builds delivery-vulnerability profiles for
        batters, then finds players with similar responses to:

        ```text
        phase + bowler type + bowler arm + length + line + over/round angle
        ```

        For every such context, the cosine-similarity vector compares these
        feature types:

        ```text
        smoothed runs/ball delta
        smoothed dot-rate delta
        smoothed boundary-rate delta
        smoothed wicket-rate delta
        ```

        Delta means the batter's result against that delivery context compared
        with that same batter's own overall baseline. So the model is comparing
        vulnerability patterns, not just whether two batters have similar
        overall strike rates.

        The reason is practical: if the batter's own sample is thin, similar
        batters add evidence without falling back to completely generic data.
        The comparison uses cosine similarity, which compares the shape of a
        batter's strengths and weaknesses rather than only overall strike rate.

        **2. Sequence lift mining**

        Data Science technique:

        ```text
        sequential pattern mining + lift analysis
        ```

        This is used because bowling plans are not always independent
        one-ball decisions. Some wicket balls work better after a setup ball.
        The app mines historical 2-ball and 3-ball patterns inside the same
        over, then checks whether the pattern has genuine lift.

        The app compares:

        ```text
        wicket rate after setup sequence
        vs
        normal wicket rate of the final delivery
        ```

        Formula:

        ```text
        lift = sequence wicket rate / final-delivery baseline wicket rate
        ```

        A sequence is only considered if it has at least
        `{MIN_SEQUENCE_ATTEMPTS}` attempts, `{MIN_SEQUENCE_WICKETS}` wickets,
        and `{MIN_SEQUENCE_LIFT}`x lift. The reason is to avoid crediting a
        setup when the final ball was already a strong wicket option by itself.

        This was chosen because the cricket question is sequential:

        ```text
        Did the previous ball make this wicket ball more effective?
        ```

        Raw wicket counts cannot answer that. Lift analysis can, because it
        compares the sequence against the final-ball baseline. That is why this
        is not added for show; it directly supports setup-ball planning.

        **3. Bowler execution adjustment**

        Data Science technique:

        ```text
        statistical performance profiling + empirical reliability adjustment
        ```

        This is used because a good batter matchup is not always a good bowling
        option for the selected bowler. The app profiles how that bowler
        historically executes each length + line + angle using:

        ```text
        bowler dot%, boundary%, wicket%, contain%, error%, and runs/ball
        ```

        These rates are smoothed toward the broader bowler-type context before
        they are used. That protects the app from overreacting to a small
        sample for one bowler.

        This was chosen because the recommendation should satisfy two
        conditions:

        ```text
        the batter is vulnerable to it
        and
        the selected bowler can execute it
        ```

        Bowler execution can adjust the final score, but its influence is
        capped at `{BOWLER_EXECUTION_WEIGHT:.0%}` so it does not override the
        batter matchup evidence completely.

        This is not a heavy machine-learning model. It is applied statistics:
        the app estimates a bowler's historical execution reliability and uses
        it as a controlled adjustment to the batter-based recommendation.
        """)


def render_reliability_rules():
    with st.expander("Reliability and minimum-evidence rules", expanded=False):
        st.markdown(f"""
        **Composite delivery type** means the same length + line + over/round angle.

        **Plan eligibility**

        A delivery can enter the six-ball plan if it has:

        ```text
        raw balls >= {MIN_RELIABLE}
        OR
        raw wickets >= 2
        ```

        **Table reliability marker**

        The checkmark in the table uses effective balls:

        ```text
        effective balls >= {MIN_RELIABLE}
        OR
        raw wickets >= 2
        ```

        Raw balls are shown in the table. Effective balls are used for scoring
        confidence after recency weighting.
        """)


def render_data_used_panel(
    batter: str,
    phase: str,
    bowler_type: str,
    bowler_hand: str,
    stadium: str,
    data_source: str,
    summary: dict,
    matchup_status: dict,
    similar_batsmen_used: list,
    dtable: pd.DataFrame,
):
    bowler_exec_used = (
        "bowler_exec_balls" in dtable.columns
        and dtable["bowler_exec_balls"].notna().any()
    )
    matchup_msg = matchup_status.get("message", "No direct matchup selected.")
    if matchup_status.get("state") == "not_selected":
        matchup_msg = "Not selected"
    matchup_state = matchup_status.get("state")
    if matchup_state == "used":
        matchup_decision = "Full matchup: direct bowler-vs-batter table replaced the broader table."
    elif matchup_state == "partial":
        matchup_decision = "Partial matchup: broader table stayed as the base; matching delivery rows were adjusted."
    elif matchup_state in ("sparse", "none"):
        matchup_decision = "Not used: direct matchup evidence did not pass full or partial-matchup rules."
    else:
        matchup_decision = "Not selected"
    similar_msg = (
        f"Used: {len(similar_batsmen_used)} similar batter(s)"
        if similar_batsmen_used else "Not used"
    )

    with st.expander("Data used for this recommendation", expanded=True):
        st.markdown(f"""
        - **Batter:** {batter}
        - **Phase:** {phase}
        - **Bowler type:** {bowler_hand} arm {bowler_type}
        - **Ground filter:** {stadium or "All grounds"}
        - **Data source:** {data_source}
        - **Raw balls used:** {summary["balls"]}
        - **Effective balls after recency weighting:** {summary["effective_balls"]}
        - **Direct matchup:** {matchup_msg}
        - **Direct matchup decision:** {matchup_decision}
        - **Similar batters:** {similar_msg}
        - **Bowler execution adjustment:** {"Used" if bowler_exec_used else "Not used"}
        """)

        with st.expander("Direct matchup rules", expanded=False):
            st.markdown(f"""
            The app has three possible direct-matchup outcomes:

            ```text
            Full matchup     -> direct matchup replaces the broader table
            Partial matchup  -> broader table remains base; direct matchup adjusts matching deliveries
            Not used         -> direct matchup is too thin or not meaningfully different
            ```

            **Complete direct matchup is used only when it is strong enough to
            replace the broader table.**

            Conditions:

            ```text
            selected bowler vs selected batter exists
            AND direct matchup table has at least 2 reliable delivery types
            ```

            A reliable delivery type means:

            ```text
            same length + line + over/round angle
            AND
            effective balls >= {MIN_RELIABLE}
            OR
            raw wickets >= 2
            ```

            When this condition passes, the recommendation table uses the
            direct bowler-vs-batter matchup as the main source.

            **Partial direct matchup is used when the direct matchup is useful
            but not strong enough to replace the base table.**

            Conditions:

            ```text
            complete direct matchup did not pass
            AND selected bowler vs selected batter has at least {PARTIAL_MATCHUP_MIN_TOTAL_BALLS} total balls
            AND same delivery type has at least {PARTIAL_MATCHUP_MIN_DELIVERY_BALLS} balls or {PARTIAL_MATCHUP_MIN_DELIVERY_WICKETS} wicket
            AND direct matchup differs meaningfully from the broader table
            ```

            Meaningfully different means at least one of:

            ```text
            SR difference >= {PARTIAL_MATCHUP_SR_DIFF}
            dot-ball% difference >= {PARTIAL_MATCHUP_DOT_DIFF} percentage points
            boundary% difference >= {PARTIAL_MATCHUP_BOUNDARY_DIFF} percentage points
            direct wicket evidence exists
            ```

            Partial matchup never replaces the base table. It only makes a
            capped adjustment:

            ```text
            maximum direct matchup influence = {PARTIAL_MATCHUP_MAX_WEIGHT:.0%}
            ```

            This keeps the plan stable while still using useful selected
            bowler-vs-batter evidence.
            """)


def render_recency_weighting_methodology(min_year=None, max_year=None):
    year_range = (
        f"{int(min_year)}-{int(max_year)}"
        if min_year is not None and max_year is not None
        else "the uploaded seasons"
    )
    with st.expander("How season-wise recency weighting works", expanded=False):
        st.markdown(f"""
        Recency weighting makes recent seasons count more than older seasons.

        For {year_range}, the latest year gets full weight:

        ```text
        latest season weight = 1.00
        one year older       = {RECENCY_DECAY_PER_YEAR:.2f}
        two years older      = {RECENCY_DECAY_PER_YEAR:.2f} x {RECENCY_DECAY_PER_YEAR:.2f}
        minimum weight       = {MIN_RECENCY_WEIGHT:.2f}
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
        older balls keep decreasing, but never below 0.30
        ```

        So 10 actual balls from an older season may count as fewer than 10
        effective balls in rates, smoothing, reliability checks, similar
        batter profiles, sequence mining, and bowler execution.
        """)


def render_score_methodology(intent: str):
    if intent == "CONTAIN":
        with st.expander("How CONTAIN score is calculated", expanded=False):
            st.markdown(f"""
            The table is ranked by a containment score. A higher score means
            the delivery is better at limiting scoring for this batter in
            this context.

            **CONTAIN score uses:**

            - **Contained-ball rate:** how often this delivery keeps the batter to 0, 1, or 2 runs.
            - **Smoothed SR:** lower scoring rate improves the score.
            - **Wicket bonus:** wickets help, but count at half-weight because the main goal is containment.
            - **Effective balls:** older seasons count less, so recent evidence has more influence.
            - **Confidence:** delivery types with fewer than `{MIN_RELIABLE}` effective balls are reduced.

            Formula:

            ```text
            raw score =
              0.80 x smoothed contained-ball%
            + 0.20 x low-SR value
            + 0.50 x wicket bonus

            final score = raw score x confidence
            ```

            Confidence:

            ```text
            confidence = min(1.0, effective balls / {MIN_RELIABLE})
            ```

            So a delivery with only 5 effective balls can only receive about
            half of its raw score, even if its early numbers look strong.

            **How smoothing works**

            Raw percentages can overreact when a delivery has only a few balls.
            Smoothing blends the delivery's own result with a prior. When the
            table is phase-specific, the prior is the batter's all-phase record
            against the same length + line + over/round angle, if available.
            If that same-delivery prior is unavailable, the app falls back to
            the wider context average for this batter and bowler type.

            ```text
            smoothed rate =
              (delivery success count + context rate x prior balls)
              / (delivery effective balls + prior balls)
            ```

            Example: suppose a phase-specific delivery has 2 wickets in 5
            effective balls. The raw wicket% is 40%, but that is too unstable.
            If this same delivery has a 4% wicket rate across all phases and
            wicket smoothing uses 60 prior balls:

            ```text
            smoothed wicket rate =
              (2 + 0.04 x 60) / (5 + 60)
              = 4.4 / 65
              = 6.8%
            ```

            So the delivery still gets credit for taking wickets, but it is
            not treated like a true 40% wicket option from only 5 balls.

            As sample size grows, the delivery's own data gets more weight.
            With a small sample, the context average protects the ranking from
            overreacting.
            """)

    elif intent == "WICKET":
        with st.expander("How WICKET score is calculated", expanded=False):
            st.markdown(f"""
            The table is ranked by a wicket-taking score. A higher score means
            the delivery has stronger evidence for creating wicket pressure
            against this batter in this context.

            **WICKET score uses:**

            - **Smoothed wicket%:** main driver of the score.
            - **Smoothed dot%:** dots help build pressure before a mistake.
            - **Smoothed SR:** lower scoring rate improves the score.
            - **Wicket bonus:** rewards deliveries with actual wicket evidence.
            - **Effective balls:** older seasons count less, so recent evidence has more influence.
            - **Confidence:** delivery types with fewer than `{MIN_RELIABLE}` effective balls are reduced.

            Formula:

            ```text
            raw score =
              0.60 x smoothed wicket%
            + 0.20 x smoothed dot%
            + 0.20 x low-SR value
            + wicket bonus

            final score = raw score x confidence
            ```

            Wicket bonus:

            ```text
            wicket bonus = min(weighted wickets x 5, 20) x confidence
            ```

            **How smoothing works**

            Raw percentages can overreact when a delivery has only a few balls.
            Smoothing blends the delivery's own result with a prior. When the
            table is phase-specific, the prior is the batter's all-phase record
            against the same length + line + over/round angle, if available.
            If that same-delivery prior is unavailable, the app falls back to
            the wider context average for this batter and bowler type.

            ```text
            smoothed rate =
              (delivery success count + context rate x prior balls)
              / (delivery effective balls + prior balls)
            ```

            Example: suppose a phase-specific delivery has 2 wickets in 5
            effective balls. The raw wicket% is 40%, but that is too unstable.
            If this same delivery has a 4% wicket rate across all phases and
            wicket smoothing uses 60 prior balls:

            ```text
            smoothed wicket rate =
              (2 + 0.04 x 60) / (5 + 60)
              = 4.4 / 65
              = 6.8%
            ```

            So the delivery still gets credit for taking wickets, but it is
            not treated like a true 40% wicket option from only 5 balls.

            As sample size grows, the delivery's own data gets more weight.
            With a small sample, the context average protects the ranking from
            overreacting.
            """)


def render_delivery_table(dtable: pd.DataFrame, intent: str):
    if dtable.empty:
        st.caption("No composite delivery data available.")
        return
    display = dtable.copy()
    display["score"] = display.apply(
        lambda r: score_delivery(r.to_dict(), intent), axis=1
    )
    display = display.sort_values("score", ascending=False).head(15)

    cols_show = [c for c in DELIVERY_COLS if c in display.columns]
    has_bowler_exec = "bowler_exec_balls" in display.columns
    has_partial_matchup = (
        "partial_matchup_used" in display.columns
        and display["partial_matchup_used"].fillna(False).any()
    )
    rows = ""
    for _, row in display.iterrows():
        rel  = "✓" if row["reliable"] else "⚠︎"
        rel_cls = "" if row["reliable"] else 'class="r"'
        sr_cls  = 'class="r"' if row["sr"] > 150 else ('class="g"' if row["sr"] < 100 else "")
        wk_cls  = 'class="g"' if row["wkt_pct"] > 8 else ""
        fp = ", ".join(row["top_fields"][:3]) if row["top_fields"] else "—"
        fp = html.escape(str(fp))
        sm_wkt = row.get("smoothed_wkt_pct", row["wkt_pct"])
        sm_dot = row.get("smoothed_dot_pct", row["dot_pct"])
        bowler_cells = ""
        if has_bowler_exec and pd.notna(row.get("bowler_exec_balls")):
            bowler_cells = (
                f"<td>{int(row['bowler_exec_balls'])}</td>"
                f"<td>{row.get('bowler_smoothed_dot_pct', '—')}%</td>"
                f"<td>{row.get('bowler_smoothed_boundary_pct', '—')}%</td>"
                f"<td>{row.get('bowler_smoothed_error_pct', '—')}%</td>"
            )
        elif has_bowler_exec:
            bowler_cells = "<td>—</td><td>—</td><td>—</td><td>—</td>"

        matchup_cells = ""
        if has_partial_matchup:
            if bool(row.get("partial_matchup_used", False)):
                delta = _num(row.get("partial_matchup_score_delta", 0))
                direction = str(row.get("partial_matchup_direction", "adjusted"))
                weight = _num(row.get("partial_matchup_weight", 0))
                balls = int(_num(row.get("partial_matchup_balls", 0)))
                matchup_cells = (
                    f'<td style="font-size:0.7rem;color:#6b7280;">'
                    f'Score {direction} {delta:+.2f}<br>'
                    f'<span>{balls} direct balls · {weight:.0%} influence</span></td>'
                )
            else:
                matchup_cells = "<td>—</td>"

        key_cells = "".join(
            f"<td>{html.escape(str(row.get(c, '—')))}</td>" for c in cols_show
        )
        rows += (
            "<tr>"
            f"{key_cells}"
            f"<td>{int(row['runs'])}</td>"
            f"<td {rel_cls}>{row['balls']} {rel}</td>"
            f"<td {sr_cls}>{row['sr']}</td>"
            f"<td {wk_cls}>{int(row['wickets'])}</td>"
            f"<td>{row['wkt_pct']}%</td>"
            f"<td>{row['dot_pct']}%</td>"
            f"<td>{sm_wkt}%</td>"
            f"<td>{sm_dot}%</td>"
            f"{bowler_cells}"
            f"{matchup_cells}"
            f'<td style="color:#6b7280;font-size:0.72rem;">{fp}</td>'
            "</tr>"
        )

    header = "".join(f"<th>{c.replace('TypeId','').replace('Id','').replace('bowling','')}</th>"
                     for c in cols_show)
    bowler_header = (
        "<th>Bowler balls</th><th>Bowler dot%</th><th>Bowler boundary%</th><th>Bowler error%</th>"
        if has_bowler_exec else ""
    )
    matchup_header = "<th>Selected bowler vs batter effect</th>" if has_partial_matchup else ""
    score_label = (
        "Sorted by Bayesian-smoothed batter score, adjusted for bowler execution "
        "and partial direct matchup when available"
    )
    table_html = (
        '<div style="overflow-x:auto;width:100%;max-width:100%;">'
        '<table class="htable" style="min-width:920px;">'
        f'<tr>{header}<th>Runs</th><th>Balls</th><th>SR</th><th>Wkts</th>'
        f'<th>Wkt%</th><th>Dot%</th><th>Smoothed Wicket%</th><th>Smoothed Dot%</th>'
        f'{bowler_header}{matchup_header}<th>Suggested fielders</th></tr>'
        f'{rows}'
        '</table></div>'
        f'<p style="font-size:0.7rem;color:#9ca3af;margin-top:4px;">'
        f'✓ ≥{MIN_RELIABLE} effective balls (reliable) &nbsp;|&nbsp; '
        f'⚠︎ &lt;{MIN_RELIABLE} effective balls (small sample)'
        f'&nbsp;|&nbsp; Effective balls = actual balls after recency weighting: '
        f'latest season counts 1.00, each older season counts '
        f'{RECENCY_DECAY_PER_YEAR:.2f}x of the next newer season, '
        f'floor {MIN_RECENCY_WEIGHT:.2f}. '
        f'Older data still contributes, but recent data has more influence. '
        f'&nbsp;|&nbsp; {score_label}</p>'
    )
    if has_partial_matchup:
        table_html += (
            '<p style="font-size:0.7rem;color:#6b7280;margin-top:2px;">'
            'Selected bowler vs batter effect = a capped score change from exact '
            'head-to-head evidence. It does not replace the broader table; it only '
            'nudges matching delivery types when the direct matchup is meaningfully different.'
            '</p>'
        )
    st.markdown(table_html, unsafe_allow_html=True)


def render_score_breakdown(dtable: pd.DataFrame, intent: str):
    if dtable.empty:
        return

    display = dtable.copy()
    display["score"] = display.apply(
        lambda r: score_delivery(r.to_dict(), intent), axis=1
    )
    display = display.sort_values("score", ascending=False).head(5)

    with st.expander("Why the top deliveries rank highest", expanded=False):
        for _, row in display.iterrows():
            bd = score_delivery_breakdown(row.to_dict(), intent)
            title = (
                f"{row.get('lengthTypeId', '—')} · "
                f"{row.get('lineTypeId', '—')} · "
                f"{row.get('bowlingFromId', '—')}"
            )
            st.markdown(f"**{title}**")
            if intent == "WICKET":
                st.caption(
                    f"Wicket component {bd['wicket_component']:.2f} · "
                    f"Dot component {bd['dot_component']:.2f} · "
                    f"Low-SR component {bd['low_sr_component']:.2f} · "
                    f"Wicket bonus {bd['wicket_bonus']:.2f}"
                )
            else:
                st.caption(
                    f"Contain component {bd['contain_component']:.2f} · "
                    f"Low-SR component {bd['low_sr_component']:.2f} · "
                    f"Wicket bonus {bd['wicket_bonus']:.2f}"
                )
            bowler_text = (
                f" · Bowler score {bd['bowler_score']:.2f}"
                if bd["bowler_score"] is not None else ""
            )
            st.caption(
                f"Confidence {bd['confidence']:.2f} · "
                f"Empirical score {bd['empirical_score']:.2f} · "
                f"Bowler weight {bd['bowler_weight']:.2f}"
                f"{bowler_text} · Final score {bd['final_score']:.2f}"
            )
            if bool(row.get("partial_matchup_used", False)):
                st.caption(
                    "Selected bowler-vs-batter effect: "
                    f"{row.get('partial_matchup_direction', 'adjusted')} by "
                    f"{_num(row.get('partial_matchup_score_delta', 0)):+.2f} "
                    "score points because this exact bowler-vs-batter record "
                    "differs from the broader data. "
                    f"Used {int(_num(row.get('partial_matchup_balls', 0)))} "
                    f"direct balls with {_num(row.get('partial_matchup_weight', 0)):.0%} "
                    f"influence. Reason: {row.get('partial_matchup_reason', '')}"
                )


def render_similar_batsmen(similar_info: list):
    """Render a box showing which similar batsmen were used and why."""
    if not similar_info:
        return
    lines = []
    names = []
    for item in similar_info:
        if len(item) == 3:
            name, similarity, prof = item
            sim_text = f", similarity {round(float(similarity) * 100, 1)}%"
        else:
            name, prof = item
            sim_text = ""
        names.append(str(name))
        lines.append(
            f"<b>{name}</b>{sim_text} — SR {prof['sr']}, Bdry% {prof['boundary_pct']}%, "
            f"Dot% {prof['dot_pct']}%, Death SR {prof['death_sr']}"
        )
    st.info(f"Similar batter supplement used: {', '.join(names)}")
    st.markdown(f"""
    <div class="similar-box">
      <strong>🔍 Supplemented with similar batsmen</strong> (batter's own data was thin)<br>
      {"<br>".join(lines)}
    </div>""", unsafe_allow_html=True)


def render_plan_summary(plan: dict):
    balls = plan.get("balls", [])
    if not balls:
        return

    role_counts = Counter(b.get("role", "best") for b in balls)
    bowler_adjusted = sum(1 for b in balls if _has_bowler_execution(b.get("stats", {})))
    small_samples = sum(
        1 for b in balls if not b.get("stats", {}).get("reliable", False)
    )
    partial_matchup_count = sum(
        1 for b in balls if b.get("stats", {}).get("partial_matchup_used", False)
    )
    seq = plan.get("sequence_meta")
    danger = plan.get("danger", {})
    danger_from = {
        "Over": "Over the wicket",
        "Round": "Round the wicket",
    }.get(danger.get("from", ""), danger.get("from", "—"))
    danger_used = bool(plan.get("danger_used"))
    danger_count = int(plan.get("danger_count", 0))
    matchup_status = plan.get("matchup_status", {})

    lines = [
        f"- **Base picks:** {role_counts.get('best', 0)} ball(s) selected from highest {plan['intent']} score.",
        f"- **Repeat control:** delivery choices respect the max-repeat cap of {MAX_REPEATS}.",
        f"- **Bowler execution:** adjusted {bowler_adjusted} ball(s) using selected bowler reliability.",
    ]

    if matchup_status.get("message") and matchup_status.get("state") != "not_selected":
        lines.append(f"- **Matchup data:** {matchup_status['message']}")

    if partial_matchup_count:
        lines.append(
            f"- **Partial matchup adjustment:** {partial_matchup_count} planned ball(s) "
            "were adjusted using selected bowler-vs-batter evidence while keeping "
            "the broader table as the base."
        )

    if seq:
        lines.append(
            f"- **Sequence pattern:** used because setup lift was {seq['lift']}x "
            f"({seq['sequence_wicket_rate']}% after setup vs "
            f"{seq['final_wicket_rate']}% final-ball baseline)."
        )
        if plan.get("intent") == "WICKET":
            lines.append(
                "- **Sequence placement:** WICKET intent uses the setup pair early "
                "on balls 1-2 because the plan is trying to attack immediately."
            )
        elif plan.get("phase") == "Death":
            lines.append(
                "- **Sequence placement:** death-over containment usually avoids "
                "forced setup patterns because execution and risk control matter more."
            )
        else:
            lines.append(
                "- **Sequence placement:** CONTAIN intent keeps balls 1-4 as "
                "best-score pressure balls, then uses ball 5 as the setup and "
                "ball 6 as the wicket ball."
            )
    else:
        lines.append(
            "- **Sequence pattern:** no high-lift setup pattern was strong enough to force into the plan."
        )

    if danger_used:
        lines.append(
            f"- **Danger warning:** {danger.get('length', '—')} · "
            f"{danger.get('line', '—')} · from {danger_from} still appears "
            f"{danger_count} ball(s) "
            f"because the planner did not find enough stronger non-danger alternatives "
            f"after reliability and repeat filters. Raw SR {danger.get('sr', '—')} "
            f"from {danger.get('balls', '—')} balls."
        )
    else:
        lines.append(
            f"- **Danger avoided:** {danger.get('length', '—')} · "
            f"{danger.get('line', '—')} · from {danger_from} had raw SR "
            f"{danger.get('sr', '—')} from {danger.get('balls', '—')} balls."
        )

    if small_samples:
        lines.append(
            f"- **Sample warning:** {small_samples} planned ball(s) rely on small effective samples."
        )

    with st.expander("Why this over plan was built this way", expanded=False):
        st.markdown("\n".join(lines))

        with st.expander("How Never Bowl is decided", expanded=False):
            st.markdown(f"""
            **Never Bowl** is the danger-zone delivery: the delivery type where
            the batter has shown the strongest scoring threat.

            The app first looks for delivery types with enough effective
            evidence:

            ```text
            effective balls >= {MIN_RELIABLE}
            ```

            If no delivery has that much effective evidence, the app falls back
            to all eligible delivery rows.

            It then selects the danger delivery using:

            ```text
            highest smoothed SR
            then higher boundary%
            then higher effective balls
            ```

            The UI displays raw SR and raw balls because they are easier to
            interpret. The selection uses smoothed SR so a tiny sample does not
            create an exaggerated Never Bowl warning.

            Current danger delivery:

            ```text
            {danger.get('length', '—')} + {danger.get('line', '—')} + from {danger_from}
            raw SR: {danger.get('sr', '—')}
            raw balls: {danger.get('balls', '—')}
            ```

            If this delivery still appears in the plan, the UI marks it as a
            high-risk fallback rather than claiming it was avoided.
            """)


def render_matchup_status(matchup_status: dict):
    message = matchup_status.get("message")
    if not message:
        return

    state = matchup_status.get("state")
    if state == "not_selected":
        return
    if state == "used":
        st.success(message)
    elif state == "partial":
        st.info(message)
    elif state == "sparse":
        st.warning(message)
    elif state == "none":
        st.info(message)
    else:
        st.info(message)


def render_glossary():
    with st.expander("Glossary", expanded=False):
        st.markdown("""
        - **Raw balls:** actual legal balls in the dataset.
        - **Effective balls:** raw balls after season-wise recency weighting.
        - **Smoothed Wicket%:** wicket rate pulled toward a prior to avoid small-sample overreaction.
        - **Smoothed Dot%:** dot rate pulled toward a prior.
        - **Contain%:** percentage of balls where the batter scored 0, 1, or 2.
        - **Direct matchup:** selected bowler vs selected batter.
        - **Similar batter supplement:** extra data from batters with similar delivery vulnerabilities.
        - **Bowler execution:** how well the selected bowler executes that delivery type historically.
        - **Sequence lift:** how much better a setup sequence performs versus the final ball alone.
        - **Danger zone:** the delivery type where the batter scores most freely.
        """)


def render_plan(plan: dict, narration: dict, batter: str, context_tag: str,
                similar_info: list | None = None):
    st.markdown(f"""
    <div class="hdr">
      <div class="bname">vs {batter}</div>
      <div class="ctag">{context_tag}</div>
    </div>
    <div class="warn-box" style="margin-bottom:10px;">
      <strong>Intent:</strong> {plan['intent']} &nbsp;·&nbsp; {plan['intent_reason']}
      {"&nbsp;·&nbsp; ⚠︎ All recommendations from small samples" if plan.get('used_unreliable') else ""}
      {"&nbsp;·&nbsp; 🔗 Sequence pattern detected" if plan.get('has_sequence_pattern') else ""}
    </div>""", unsafe_allow_html=True)

    render_matchup_status(plan.get("matchup_status", {}))
    render_plan_summary(plan)

    seq = plan.get("sequence_meta")
    if seq:
        st.caption(
            "Sequence evidence: "
            f"{seq['lift']}x lift · "
            f"{seq['sequence_wicket_rate']}% wicket rate after setup vs "
            f"{seq['final_wicket_rate']}% for the final delivery alone · "
            f"{seq['attempts']} attempts"
        )

    # Show similar batsmen info if used
    if similar_info:
        render_similar_batsmen(similar_info)

    intent_cls = {
        "Wicket ball": "t-wicket",
        "Contain":     "t-contain",
        "Set up":      "t-setup",
        "Surprise":    "t-surprise",
    }

    why_map = {}
    if narration and "ball_explanations" in narration:
        for item in narration["ball_explanations"]:
            why_map[item["ball"]] = item.get("why", "")

    for b in plan["balls"]:
        s       = b["stats"]
        cls     = intent_cls.get(b["intent"], "t-contain")
        why_txt = why_map.get(b["ball"], "")
        fields  = ", ".join(b["fields"]) if b["fields"] else "Insufficient data for field placement"
        rel_tag = ('<span class="tag t-warn">⚠︎ Small sample</span>'
                   if not s["reliable"] else "")
        role_label = b.get("role", "")

        primary = f"{b['length']} · {b['line']}"
        # Sub-line: from_wicket + hand (variation handled separately below)
        _from_map = {"Over": "Over the wicket", "Round": "Round the wicket"}
        sub_parts = []
        for val, label in [(_from_map.get(b["from_wicket"], b["from_wicket"]), ""), (b["hand"], "arm")]:
            if val and val not in ("—", "Unknown", "nan"):
                sub_parts.append(f"{val} {label}".strip())
        secondary = " · ".join(sub_parts)

        # Variation block
        vs = b.get("variation_stats", {})
        var_name = b.get("variation", "—")
        if var_name and var_name not in ("—", "Unknown", "nan") and vs:
            var_html = (
                f'<div style="margin:6px 0 8px;padding:6px 10px;background:#f8fafc;'
                f'border-left:3px solid #6366f1;border-radius:0 6px 6px 0;font-size:0.8rem;">'
                f'<div style="font-weight:600;color:#4f46e5;margin-bottom:3px;">'
                f'Why this variation was chosen: {var_name}</div>'
                f'<div style="color:#6b7280;">'
                f'Observed variation record within this length + line: '
                f'wicket rate {vs.get("wkt_pct","?")}%, '
                f'dot-ball rate {vs.get("dot_pct","?")}%, '
                f'SR conceded {vs.get("sr","?")}, '
                f'from {vs.get("balls","?")} raw balls. '
                f'This chooses the best variation after the main length + line has already been selected.'
                f'</div>'
            )
            var_html += '</div>'
        else:
            var_html = ""

        bowler_html = ""
        if _has_bowler_execution(s):
            bowler_html = (
                f'<div style="font-size:0.76rem;color:#6b7280;margin:6px 0 8px;'
                f'padding:6px 10px;background:#f9fafb;border-radius:6px;">'
                f'<strong>Selected bowler reliability:</strong> '
                f'smoothed dot-ball rate {s["bowler_smoothed_dot_pct"]}% · '
                f'boundary rate {s["bowler_smoothed_boundary_pct"]}% · '
                f'error rate {s["bowler_smoothed_error_pct"]}% · '
                f'sample {int(s.get("bowler_exec_balls", 0))} balls'
                f'</div>'
            )

        matchup_html = ""
        if s.get("partial_matchup_used"):
            delta = _num(s.get("partial_matchup_score_delta", 0))
            direction = s.get("partial_matchup_direction", "adjusted")
            weight = _num(s.get("partial_matchup_weight", 0))
            balls_used = int(_num(s.get("partial_matchup_balls", 0)))
            matchup_html = (
                f'<div style="font-size:0.76rem;color:#6b7280;margin:6px 0 8px;'
                f'padding:6px 10px;background:#fff7ed;border-radius:6px;">'
                f'<strong>Selected bowler-vs-batter effect:</strong> '
                f'this delivery score was {direction} by {delta:+.2f} because the exact '
                f'bowler-vs-batter history differed from the broader matchup data. '
                f'The app used {balls_used} direct balls with only {weight:.0%} influence, '
                f'so the broader data still controls most of the ranking. '
                f'Reason: {html.escape(str(s.get("partial_matchup_reason", "")))}'
                f'</div>'
            )

        evidence_html = (
            f'<div style="font-size:0.76rem;color:#6b7280;margin:8px 0;line-height:1.55;">'
            f'<strong>Why this length + line was chosen:</strong><br>'
            f'Historical record for this length + line: SR conceded {s["sr"]} from {s["balls"]} raw balls.<br>'
            f'After smoothing: wicket chance '
            f'{s.get("smoothed_wkt_pct", s["wkt_pct"])}%, '
            f'dot-ball chance {s.get("smoothed_dot_pct", s["dot_pct"])}%.<br>'
            f'The app ranks using smoothed numbers so small samples do not dominate.'
            f'</div>'
        )

        st.markdown(f"""
        <div class="ball-card">
          <div class="ball-num">Ball {b['ball']} {"· " + role_label.upper() if role_label else ""}</div>
          <div class="delivery">{primary}</div>
          <div class="sub-delivery">{secondary}</div>
          {var_html}
          {bowler_html}
          {matchup_html}
          <div class="why-text">{why_txt}</div>
          {evidence_html}
          <span class="tag t-field">Likely fielding zones: {fields}</span>
          <span class="tag {cls}">{b['intent']}</span>
          {rel_tag}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    overall = narration.get("overall_note", "") if narration else ""
    danger_txt = narration.get("danger_explanation", "") if narration else ""
    d = plan["danger"]
    danger_used = bool(plan.get("danger_used"))
    danger_count = int(plan.get("danger_count", 0))
    danger_from = {
        "Over": "Over the wicket",
        "Round": "Round the wicket",
    }.get(d.get("from", ""), d.get("from", "—"))
    danger_hand = d.get("hand", "—")
    danger_hand_text = (
        f" · {danger_hand} arm"
        if danger_hand not in ("—", "Unknown", "nan", None, "")
        else ""
    )

    if danger_used:
        danger_heading = "⚠︎ High-risk delivery still used:"
        danger_note = (
            f"This delivery appears {danger_count} time(s) in the plan because "
            "the available non-danger alternatives were weaker or too thin. "
            "Treat it as a fallback, not a preferred option."
        )
    else:
        danger_heading = "🚫 Never bowl:"
        danger_note = danger_txt

    if overall:
        st.markdown(f"""
        <div class="insight-box">
          <strong>⚡ Data pattern:</strong> {overall}
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="danger-box">
      <strong>{danger_heading}</strong>
      {d['length']} · {d['line']} · from {danger_from}{danger_hand_text}
      — raw SR {d['sr']}, {d['balls']} balls.<br>
      {danger_note}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("## T20 Match-day bowling planner")
    st.markdown(
        "<p style='color:#888;margin-top:-0.4rem;margin-bottom:1.4rem;font-size:0.87rem;'>"
        "Dataset contains ball-by-ball data from 2019 onwards.<br>"
        "The over plan is generated by deterministic logic; the LLM is only used "
        "for narration of the over plan."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load data from Hugging Face ───────────────────────────────────────────
    required_hf_secrets = ["HF_TOKEN", "HF_REPO_ID", "HF_FILENAME"]
    missing_hf_secrets = [k for k in required_hf_secrets if k not in st.secrets]
    if missing_hf_secrets:
        st.error(
            "Missing Hugging Face Streamlit secrets: "
            + ", ".join(missing_hf_secrets)
        )
        st.info(
            "Add HF_TOKEN, HF_REPO_ID, and HF_FILENAME in Streamlit secrets "
            "before deploying."
        )
        return

    dataset_signature = (
        st.secrets["HF_REPO_ID"],
        st.secrets["HF_FILENAME"],
    )
    if (
        "df" not in st.session_state
        or st.session_state.get("dataset_signature") != dataset_signature
    ):
        df = load_huggingface_data(
            st.secrets["HF_REPO_ID"],
            st.secrets["HF_FILENAME"],
            st.secrets["HF_TOKEN"],
        )
        st.session_state["df"] = df
        st.session_state["dataset_signature"] = dataset_signature
        st.session_state["profiles"] = build_batter_profiles(df)

    df = st.session_state["df"]
    with st.expander("Dataset loaded", expanded=False):
        st.success(
            f"✓ {len(df):,} balls · "
            f"{df['battingPlayer'].nunique()} batters · "
            f"{df['fixtureId'].nunique()} matches"
        )
        if DATA_YEAR_COL in df.columns and df[DATA_YEAR_COL].notna().any():
            st.caption(
                f"Recency weighting active: {int(df[DATA_YEAR_COL].min())}-"
                f"{int(df[DATA_YEAR_COL].max())}, "
                f"{RECENCY_DECAY_PER_YEAR:.0%} influence retained per older year "
                f"(floor {MIN_RECENCY_WEIGHT:.0%})."
            )
        else:
            st.caption("Recency weighting inactive: no usable year column found.")

    df_all: pd.DataFrame = st.session_state["df"]
    if DATA_YEAR_COL in df_all.columns and df_all[DATA_YEAR_COL].notna().any():
        render_recency_weighting_methodology(
            df_all[DATA_YEAR_COL].min(),
            df_all[DATA_YEAR_COL].max(),
        )
    render_app_methodology()
    render_ds_methodology()
    render_reliability_rules()
    profiles = st.session_state.get("profiles", {})
    batters = sorted(df_all["battingPlayer"].dropna().unique())

    st.markdown("---")

    # ── Inputs ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Batter**")
        batter = st.selectbox("Batter", batters, label_visibility="collapsed")

        selected_bowler_label = st.session_state.get(
            "selected_bowler_label", "— Not selected —"
        )
        selected_bowler = (
            "" if selected_bowler_label == "— Not selected —" else selected_bowler_label
        )

        st.markdown("**Bowler type**")
        bowler_type_options = bowler_type_options_for(df_all, selected_bowler)
        if st.session_state.get("selected_bowler_type") not in bowler_type_options:
            st.session_state["selected_bowler_type"] = bowler_type_options[0]
        bowler_type = st.selectbox("Bowler type",
            bowler_type_options,
            key="selected_bowler_type",
            label_visibility="collapsed")

        st.markdown("**Bowler arm**")
        bowler_hand_options = bowler_hand_options_for(
            df_all, selected_bowler, bowler_type
        )
        if st.session_state.get("selected_bowler_hand") not in bowler_hand_options:
            st.session_state["selected_bowler_hand"] = bowler_hand_options[0]
        bowler_hand = st.selectbox(
            "Bowler arm",
            bowler_hand_options,
            key="selected_bowler_hand",
            label_visibility="collapsed",
        )

        bowler_options = bowler_options_for(df_all, bowler_type, bowler_hand)
        if selected_bowler and selected_bowler not in bowler_options:
            selected_bowler_label = "— Not selected —"
            st.session_state["selected_bowler_label"] = selected_bowler_label
            selected_bowler = ""
        if selected_bowler:
            bowler_select_options = (
                ["— Not selected —", selected_bowler]
                + [b for b in bowler_options if b != selected_bowler]
            )
        else:
            bowler_select_options = ["— Not selected —"] + bowler_options
        bowler_index = (
            bowler_select_options.index(selected_bowler_label)
            if selected_bowler_label in bowler_select_options
            else 0
        )
        st.markdown("**Bowler** *(optional — for matchup data)*")
        bowler_label = st.selectbox(
            "Bowler",
            bowler_select_options,
            index=bowler_index,
            key="selected_bowler_label",
            label_visibility="collapsed",
        )
        bowler = "" if bowler_label == "— Not selected —" else bowler_label

        st.markdown("**Stadium**")
        ground_options = ["— All grounds —"] + sorted(
            df_all["ground"].dropna().astype(str).unique().tolist()
        ) if "df" in st.session_state else ["— All grounds —"]
        stadium_sel = st.selectbox("Stadium", ground_options,
                                   label_visibility="collapsed")
        stadium = "" if stadium_sel == "— All grounds —" else stadium_sel

    with c2:
        st.markdown("**Innings**")
        innings = st.selectbox("Innings",
                               ["1st innings (setting)","2nd innings (chasing)"],
                               label_visibility="collapsed")

        st.markdown("**Over number**")
        over = st.number_input("Over", 1, 20, 16, label_visibility="collapsed")

        st.markdown("**Team runs**")
        team_score = st.text_input("Team score", "21",
                                   placeholder="e.g. 21",
                                   label_visibility="collapsed")

        st.markdown("**Wickets down**")
        wickets_down = st.number_input("Wickets down", 0, 9, 3,
                                       label_visibility="collapsed")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Batter runs**")
            batter_runs = st.number_input("Runs", 0, value=10,
                                          label_visibility="collapsed")
        with b2:
            st.markdown("**Balls faced**")
            batter_balls = st.number_input("Balls faced", 1, value=10,
                                           label_visibility="collapsed")

    balls_left = (20 - int(over) + 1) * 6

    if "chasing" in innings.lower():
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Runs needed**")
            runs_needed = st.number_input("Runs needed", 0, value=48,
                                          label_visibility="collapsed")
        with r2:
            st.markdown("**Balls left**")
            balls_left = st.number_input(
                "Balls left",
                1,
                120,
                value=balls_left,
                label_visibility="collapsed",
                help="Defaults from the selected over number, but can be edited for partial overs or unusual match situations.",
            )
    else:
        runs_needed = 0

    # ── Validate inputs ───────────────────────────────────────────────────────
    try:
        team_score_int = int(team_score)
    except ValueError:
        st.error("Team score must be a valid number.")
        return

    if batter_runs > team_score_int:
        st.error(
            f"Batter runs ({batter_runs}) cannot exceed team runs ({team_score_int})."
        )
        return

    # ── Filter data ───────────────────────────────────────────────────────────
    phase   = phase_of(over)
    lo, hi  = {"Powerplay":(1,6), "Middle":(7,16), "Death":(17,20)}[phase]
    df_bat   = df_all[df_all["battingPlayer"] == batter].copy()

    # Apply bowler type + hand filter
    df_bat_typed = filter_by_bowler_type_hand(df_bat, bowler_type, bowler_hand)

    # Also filter by stadium if selected
    if stadium and "ground" in df_bat_typed.columns:
        df_bat_ground = df_bat_typed[df_bat_typed["ground"] == stadium]
        if len(df_bat_ground) >= 15:
            df_bat_typed = df_bat_ground

    df_phase = df_bat_typed[(df_bat_typed["overNumber"] >= lo) & (df_bat_typed["overNumber"] <= hi)]

    # Cascade fallback: phase (all innings) → all data for batter (typed)
    if len(df_phase) >= 15:
        df_use = df_phase
        filter_note = f"{phase}  {bowler_hand} arm {bowler_type}"
        delivery_prior_df = df_bat_typed.copy()
    else:
        df_use = df_bat_typed
        filter_note = f"All phases · {bowler_hand} arm {bowler_type} (phase filter had < 15 balls)"
        delivery_prior_df = None

    # ── Similar batsmen fallback when data is thin ────────────────────────────
    similar_batsmen_used = []
    similar_prior_dfs = []
    if len(df_use) < MIN_SIMILAR_THRESH and profiles:
        similar = find_similar_batsmen(
            batter,
            profiles,
            n=5,
            phase=phase,
            bowler_type=bowler_type,
            bowler_hand=bowler_hand,
        )
        if similar:
            similar_dfs = []
            for sim_name, sim_dist, sim_prof in similar:
                sim_df = df_all[df_all["battingPlayer"] == sim_name].copy()
                sim_df = filter_by_bowler_type_hand(sim_df, bowler_type, bowler_hand)
                if delivery_prior_df is not None and len(sim_df) >= 10:
                    similar_prior_dfs.append(sim_df)

                # Try same phase filter first
                sim_phase = sim_df[(sim_df["overNumber"] >= lo) & (sim_df["overNumber"] <= hi)]
                if len(sim_phase) >= 10:
                    similar_dfs.append(sim_phase)
                    similar_batsmen_used.append((sim_name, sim_dist, sim_prof))
                elif len(sim_df) >= 10:
                    similar_dfs.append(sim_df)
                    similar_batsmen_used.append((sim_name, sim_dist, sim_prof))

            if similar_dfs:
                df_similar_pool = pd.concat([df_use] + similar_dfs, ignore_index=True)
                df_use = df_similar_pool
                filter_note += f" + {len(similar_batsmen_used)} similar batsmen"
                if delivery_prior_df is not None and similar_prior_dfs:
                    delivery_prior_df = pd.concat(
                        [delivery_prior_df] + similar_prior_dfs,
                        ignore_index=True,
                    )

    # ── Bowler matchup filter ─────────────────────────────────────────────────
    df_matchup = pd.DataFrame()
    matchup_status = {
        "state": "not_selected",
        "used": False,
        "message": "No bowler selected; using broader batter-vs-bowler-type data.",
    }
    if bowler:
        df_mu = df_all[
            (df_all["battingPlayer"] == batter) &
            (df_all["bowlingPlayer"] == bowler)
        ].copy()
        mu_wickets = int(df_mu.apply(is_out, axis=1).sum())
        matchup_status = {
            "state": "found",
            "used": False,
            "balls": len(df_mu),
            "wickets": mu_wickets,
            "message": (
                f"Direct matchup found: {bowler} vs {batter} "
                f"({len(df_mu)} balls, {mu_wickets} wickets)."
            ),
        }
        # If bowler has 3+ wickets vs this batter, always use all matchup data
        # regardless of delivery count
        if mu_wickets >= 3:
            df_matchup = df_mu
        elif len(df_mu) >= 6:
            # Bowler specified: do not filter by phase, use all matchup data
            df_matchup = df_mu
        else:
            matchup_status = {
                "state": "none",
                "used": False,
                "balls": len(df_mu),
                "wickets": mu_wickets,
                "message": (
                    f"Direct matchup not used: only {len(df_mu)} balls for "
                    f"{bowler} vs {batter}. A reliable delivery type means the same "
                    f"length + line + over/round angle has at least {MIN_RELIABLE} "
                    f"effective balls after recency weighting. Using broader "
                    f"batter-vs-bowler-type data."
                ),
            }

    # ── Mine wicket-taking sequences from the data ────────────────────────────
    sequences = mine_wicket_sequences(df_use)

    # ── Build delivery tables ─────────────────────────────────────────────────
    dtable         = build_delivery_table(df_use, prior_df=delivery_prior_df)
    dtable_matchup = build_delivery_table(df_matchup) if not df_matchup.empty else pd.DataFrame()

    if not dtable_matchup.empty and not dtable.empty:
        reliable_mu = dtable_matchup[dtable_matchup["reliable"]].shape[0]
        if reliable_mu >= 2:
            dtable_final = dtable_matchup
            data_source  = f"Matchup: {bowler} vs {batter} ({len(df_matchup)} balls)"
            matchup_status = {
                "state": "used",
                "used": True,
                "balls": len(df_matchup),
                "wickets": int(df_matchup.apply(is_out, axis=1).sum()),
                "message": (
                    f"Direct matchup used: {bowler} vs {batter} "
                    f"({len(df_matchup)} balls)."
                ),
            }
        else:
            dtable_final = dtable
            data_source  = f"All bowlers vs {batter} ({len(df_use)} balls) — matchup data sparse"
            matchup_status = {
                "state": "sparse",
                "used": False,
                "balls": len(df_matchup),
                "wickets": int(df_matchup.apply(is_out, axis=1).sum()),
                "message": (
                    f"Direct matchup not used: {bowler} vs {batter} has "
                    f"{len(df_matchup)} balls but fewer than 2 reliable delivery types; "
                    f"a reliable delivery type means the same length + line + "
                    f"over/round angle has at least {MIN_RELIABLE} effective balls "
                    f"after recency weighting. Using broader batter-vs-bowler-type data."
                ),
            }
    else:
        dtable_final = dtable
        data_source  = f"All bowlers vs {batter} ({len(df_use)} balls)"

    if similar_batsmen_used:
        data_source += f" (incl. {len(similar_batsmen_used)} similar batsmen)"

    # ── Match intent is needed before scoring partial matchup adjustments ─────
    batter_sr_now = sr(batter_runs, batter_balls)
    team_crr      = round(team_score_int / max(over - 1, 1), 2)
    g_phase_rr    = ground_phase_avg_rr(df_all, stadium, lo, hi) if stadium else None
    intent, intent_reason = determine_intent(
        over, innings, wickets_down, runs_needed, balls_left, batter_sr_now,
        team_crr=team_crr, ground_phase_rr=g_phase_rr,
    )

    if (
        matchup_status.get("state") == "sparse"
        and not dtable_matchup.empty
        and not dtable_final.empty
    ):
        dtable_final, partial_summary = apply_partial_matchup_adjustment(
            dtable_final,
            dtable_matchup,
            int(matchup_status.get("balls", 0)),
            intent,
        )
        if partial_summary["applied"]:
            data_source += " · partial matchup adjusted"
            matchup_status = {
                **matchup_status,
                "state": "partial",
                "used": True,
                "partial_used": True,
                "partial_adjusted_rows": partial_summary["applied"],
                "message": (
                    f"Direct matchup partially used: {partial_summary['applied']} "
                    f"delivery type(s) adjusted using {bowler} vs {batter} evidence. "
                    "Broader batter-vs-bowler-type data remains the base."
                ),
            }

    if bowler:
        bowler_exec = build_bowler_execution_table(df_all, bowler, bowler_type, bowler_hand)
        if not bowler_exec.empty:
            dtable_final = add_bowler_execution(dtable_final, bowler_exec)
            data_source += " · bowler execution adjusted"

    # ── Data preview ─────────────────────────────────────────────────────────
    dtable_scored = dtable_final

    with st.expander(
        f"📊 {batter} · {filter_note} · {len(df_use)} balls", expanded=False
    ):
        summ = summary_stats(df_use)
        render_chips(summ)
        render_data_used_panel(
            batter,
            phase,
            bowler_type,
            bowler_hand,
            stadium,
            data_source,
            summ,
            matchup_status,
            similar_batsmen_used,
            dtable_scored,
        )

        # Show similar batsmen details if used
        if similar_batsmen_used:
            render_similar_batsmen(similar_batsmen_used)

        st.markdown(
            f"**Composite delivery table** *(sorted by {intent} score)*  "
            f"<span style='font-size:0.75rem;color:#6b7280;'>Source: {data_source}</span>",
            unsafe_allow_html=True,
        )
        render_matchup_status(matchup_status)
        render_score_methodology(intent)
        with st.expander("Smoothing formulas used in this table", expanded=False):
            st.markdown(f"""
            Smoothing blends the delivery's own evidence with prior evidence so
            small samples do not dominate the ranking.

            **Rate smoothing formula**

            ```text
            smoothed rate =
              (delivery success count + prior rate x prior balls)
              / (delivery effective balls + prior balls)
            ```

            Used for:

            ```text
            Smoothed Dot%      = (dot balls + prior dot rate x {DOT_PRIOR_BALLS}) / (effective balls + {DOT_PRIOR_BALLS})
            Smoothed Boundary% = (boundaries + prior boundary rate x {BOUNDARY_PRIOR_BALLS}) / (effective balls + {BOUNDARY_PRIOR_BALLS})
            Smoothed Contain%  = (0/1/2-run balls + prior contain rate x {CONTAIN_PRIOR_BALLS}) / (effective balls + {CONTAIN_PRIOR_BALLS})
            Smoothed Wicket%   = (wickets + prior wicket rate x {WICKET_PRIOR_BALLS}) / (effective balls + {WICKET_PRIOR_BALLS})
            ```

            **Runs smoothing formula**

            ```text
            smoothed runs/ball =
              (delivery runs + prior runs/ball x {RUNS_PRIOR_BALLS})
              / (delivery effective balls + {RUNS_PRIOR_BALLS})

            smoothed SR = smoothed runs/ball x 100
            ```

            **Example**

            Suppose a delivery has:

            ```text
            5 effective balls
            2 wickets
            2 dots
            1 boundary
            3 contain balls
            8 runs
            ```

            Suppose the prior rates are:

            ```text
            prior wicket rate = 4%
            prior dot rate = 30%
            prior boundary rate = 18%
            prior contain rate = 55%
            prior runs/ball = 1.30
            ```

            Then:

            ```text
            Smoothed Wicket% =
              (2 + 0.04 x {WICKET_PRIOR_BALLS}) / (5 + {WICKET_PRIOR_BALLS})
              = {round((2 + 0.04 * WICKET_PRIOR_BALLS) / (5 + WICKET_PRIOR_BALLS) * 100, 1)}%

            Smoothed Dot% =
              (2 + 0.30 x {DOT_PRIOR_BALLS}) / (5 + {DOT_PRIOR_BALLS})
              = {round((2 + 0.30 * DOT_PRIOR_BALLS) / (5 + DOT_PRIOR_BALLS) * 100, 1)}%

            Smoothed Boundary% =
              (1 + 0.18 x {BOUNDARY_PRIOR_BALLS}) / (5 + {BOUNDARY_PRIOR_BALLS})
              = {round((1 + 0.18 * BOUNDARY_PRIOR_BALLS) / (5 + BOUNDARY_PRIOR_BALLS) * 100, 1)}%

            Smoothed Contain% =
              (3 + 0.55 x {CONTAIN_PRIOR_BALLS}) / (5 + {CONTAIN_PRIOR_BALLS})
              = {round((3 + 0.55 * CONTAIN_PRIOR_BALLS) / (5 + CONTAIN_PRIOR_BALLS) * 100, 1)}%

            Smoothed runs/ball =
              (8 + 1.30 x {RUNS_PRIOR_BALLS}) / (5 + {RUNS_PRIOR_BALLS})
              = {round((8 + 1.30 * RUNS_PRIOR_BALLS) / (5 + RUNS_PRIOR_BALLS), 2)}

            Smoothed SR =
              {round((8 + 1.30 * RUNS_PRIOR_BALLS) / (5 + RUNS_PRIOR_BALLS), 2)} x 100
              = {round((8 + 1.30 * RUNS_PRIOR_BALLS) / (5 + RUNS_PRIOR_BALLS) * 100, 1)}
            ```

            For phase-specific tables, the prior comes from the batter's
            all-phase record against the same length + line + angle when
            available. Otherwise it falls back to the broader
            batter-vs-bowler-type context average.
            """)
        render_delivery_table(dtable_scored, intent)
        render_score_breakdown(dtable_scored, intent)

        # Show sequence patterns found
        if sequences:
            st.markdown(
                f"<p style='font-size:0.75rem;color:#6b7280;margin-top:8px;'>"
                f"🔗 {len(sequences)} wicket-taking sequence(s) mined from data</p>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Generate ──────────────────────────────────────────────────────────────
    context_tag = (
        f"Over {over} · {phase} · {innings.split()[0]} · "
        f"{stadium or 'Unknown ground'}"
    )
    current_plan_signature = (
        batter,
        bowler_type,
        bowler_hand,
        bowler,
        stadium,
        innings,
        int(over),
        team_score_int,
        int(wickets_down),
        int(batter_runs),
        int(batter_balls),
        int(runs_needed),
        int(balls_left),
    )
    if (
        "plan_signature" in st.session_state
        and st.session_state["plan_signature"] != current_plan_signature
    ):
        clear_plan_state()

    go = st.button("🎯 Generate over plan", use_container_width=True, type="primary")

    if go:
        if dtable_scored.empty:
            clear_plan_state()
            st.error("Not enough data to build a plan for this batter with these filters.")
        else:
            # Step 1: deterministic plan — no LLM
            plan = build_plan(dtable_scored, intent, intent_reason, sequences, df_raw=df_use, phase=phase)

            if "error" in plan:
                clear_plan_state()
                st.error(plan["error"])
            else:
                plan["matchup_status"] = matchup_status
                # Step 2: LLM narrates only
                with st.spinner("Building plan... (data decisions made, asking AI to narrate)"):
                    try:
                        narration = narrate_plan(
                            plan, batter,
                            bowler if bowler else "Not specified",
                            bowler_type, bowler_hand,
                            context_tag,
                        )
                    except Exception as e:
                        narration = {}
                        st.warning(f"Narration failed ({e}) — showing data-only plan.")

                st.session_state["plan"]        = plan
                st.session_state["narration"]   = narration
                st.session_state["plan_batter"] = batter
                st.session_state["plan_ctx"]    = context_tag
                st.session_state["similar_info"] = similar_batsmen_used
                st.session_state["plan_signature"] = current_plan_signature

    # ── Render ────────────────────────────────────────────────────────────────
    if "plan" in st.session_state:
        st.markdown("---")
        render_plan(
            st.session_state["plan"],
            st.session_state["narration"],
            st.session_state["plan_batter"],
            st.session_state["plan_ctx"],
            st.session_state.get("similar_info", []),
        )
    render_glossary()


if __name__ == "__main__":
    main()
