import os
import base64
from math import pow
from textwrap import dedent

import pandas as pd
import requests
import streamlit as st

# ==========================
# PAGE SETUP
# ==========================

st.set_page_config(
    page_title="SBT EDGE",
    page_icon="📈",
    layout="wide"
)

# ==========================
# SETTINGS
# ==========================

YEAR = 2026
START_RATING = 1500
K_FACTOR = 40
HOME_ADVANTAGE = 73
MARGIN_DIVISOR = 12
VALID_RESULTS = ["WIN", "LOSS", "PUSH", "VOID", "DRAW"]

USER_AGENT = "Jayden AFL Predictor - jayken305@gmail.com"
TEAM_NAME_MAP = {
    "Adelaide": ["Adelaide Crows", "Adelaide"],
    "Brisbane Lions": ["Brisbane Lions", "Brisbane"],
    "Carlton": ["Carlton Blues", "Carlton"],
    "Collingwood": ["Collingwood Magpies", "Collingwood"],
    "Essendon": ["Essendon Bombers", "Essendon"],
    "Fremantle": ["Fremantle Dockers", "Fremantle"],
    "Geelong": ["Geelong Cats", "Geelong"],
    "Gold Coast": ["Gold Coast Suns", "Gold Coast"],
    "GWS Giants": ["GWS Giants", "Greater Western Sydney Giants", "Greater Western Sydney"],
    "Greater Western Sydney": ["GWS Giants", "Greater Western Sydney Giants", "Greater Western Sydney"],
    "Hawthorn": ["Hawthorn Hawks", "Hawthorn"],
    "Melbourne": ["Melbourne Demons", "Melbourne"],
    "North Melbourne": ["North Melbourne Kangaroos", "North Melbourne"],
    "Port Adelaide": ["Port Adelaide Power", "Port Adelaide"],
    "Richmond": ["Richmond Tigers", "Richmond"],
    "St Kilda": ["St Kilda Saints", "St Kilda"],
    "Sydney": ["Sydney Swans", "Sydney"],
    "West Coast": ["West Coast Eagles", "West Coast"],
    "Western Bulldogs": ["Western Bulldogs"],
}


# ==========================
# STYLING
# ==========================

def load_css():
    with open("sbt_edge_logo.png", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(2,8,18,0.88), rgba(2,8,18,0.92)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stMetric"] {{
            background: rgba(10, 16, 28, 0.80);
            border: 1px solid rgba(0, 163, 255, 0.45);
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 0 18px rgba(0,163,255,0.12);
        }}

        .stButton button {{
            background: linear-gradient(90deg, #0077ff, #00aaff);
            color: white;
            border-radius: 12px;
            border: 1px solid #00aaff;
            font-weight: bold;
            padding: 0.7rem 1.4rem;
            box-shadow: 0 0 16px rgba(0,163,255,0.25);
        }}

        .stButton button:hover {{
            border: 1px solid #66ccff;
            box-shadow: 0 0 25px rgba(0,163,255,0.55);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


load_css()

# ==========================
# API
# ==========================

def squiggle(query: str, **params):
    url = "https://api.squiggle.com.au/"
    params = {"q": query, **params}
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    return response.json()

# ==========================
# ELO MODEL
# ==========================

def expected_score(rating_a, rating_b):
    return 1 / (1 + pow(10, (rating_b - rating_a) / 400))


def build_elo(games_df):
    teams = sorted(
        set(games_df["hteam"].dropna())
        | set(games_df["ateam"].dropna())
    )

    elo = {
        team: START_RATING
        for team in teams
    }

    completed = games_df[
        games_df["complete"] == 100
    ].copy()

    completed = completed.sort_values(
        ["round", "date"]
    )

    for _, game in completed.iterrows():
        home = game["hteam"]
        away = game["ateam"]

        home_rating = elo[home] + HOME_ADVANTAGE
        away_rating = elo[away]

        home_expected = expected_score(
            home_rating,
            away_rating
        )

        if game["hscore"] > game["ascore"]:
            home_actual = 1
        elif game["hscore"] < game["ascore"]:
            home_actual = 0
        else:
            home_actual = 0.5

        margin = abs(
            game["hscore"] - game["ascore"]
        )

        margin_multiplier = max(
            1,
            margin / 30
        )

        change = (
            K_FACTOR
            * margin_multiplier
            * (home_actual - home_expected)
        )

        elo[home] += change
        elo[away] -= change

    return elo

# ==========================
# SQUIGGLE CONSENSUS
# ==========================

def get_squiggle_consensus(tips_df, game_id, home, away):
    game_tips = tips_df[
        tips_df["gameid"] == game_id
    ]

    home_votes = 0
    away_votes = 0

    for _, row in game_tips.iterrows():
        tipped_team = str(
            row.get("tip", "")
        ).strip()

        if tipped_team == home:
            home_votes += 1
        elif tipped_team == away:
            away_votes += 1

    if home_votes > away_votes:
        return home, f"{home_votes}-{away_votes}"

    if away_votes > home_votes:
        return away, f"{home_votes}-{away_votes}"

    return "No consensus", f"{home_votes}-{away_votes}"

# ==========================
# RISK + EDGE SCORE
# ==========================

def risk_rating(elo_tip, squiggle_tip, confidence):
    if squiggle_tip == "No consensus":
        return "HIGH"

    if elo_tip == squiggle_tip and confidence >= 70:
        return "LOW"

    if elo_tip == squiggle_tip and confidence >= 60:
        return "MEDIUM"

    if elo_tip == squiggle_tip:
        return "MEDIUM"

    return "HIGH"

def backtest_risk_rating(confidence):
    if confidence >= 75:
        return "LOW"

    if confidence >= 60:
        return "MEDIUM"

    return "HIGH"

def calculate_edge_score(confidence, risk):
    if risk == "LOW":
        risk_bonus = 1.5
    elif risk == "MEDIUM":
        risk_bonus = 0.5
    else:
        risk_bonus = -1.0

    edge_score = (confidence / 10) + risk_bonus

    edge_score = max(
        1,
        min(10, edge_score)
    )

    return round(edge_score, 1)

# ==========================
# ODDS API + EV FUNCTIONS
# ==========================

def fetch_afl_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds/"

    params = {
        "apiKey": api_key,
        "regions": "au",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()
    return response.json()

def break_even_probability(decimal_odds):
    if decimal_odds <= 0:
        return None

    return 1 / decimal_odds


def expected_roi(model_probability, decimal_odds):
    if decimal_odds <= 0:
        return None

    return (model_probability * decimal_odds) - 1

def calculate_profit_loss(stake, result, decimal_odds):
    result = str(result).strip().upper()

    if stake <= 0:
        return 0

    if result == "WIN":
        return stake * (decimal_odds - 1)

    if result == "LOSS":
        return -stake

    if result in ["PUSH", "VOID", "DRAW"]:
        return 0

    return 0

def clean_result(result):
    result = str(result).strip().upper()

    if result in VALID_RESULTS:
        return result

    return ""

def best_worst_summary(summary_df, label_column):
    if summary_df.empty:
        return None, None

    best_row = summary_df.sort_values(
        "ROI %",
        ascending=False
    ).iloc[0]

    worst_row = summary_df.sort_values(
        "ROI %",
        ascending=True
    ).iloc[0]

    best_text = (
        f'{best_row[label_column]} | '
        f'ROI {best_row["ROI %"]}% | '
        f'Profit ${best_row["Profit_Loss"]:.2f}'
    )

    worst_text = (
        f'{worst_row[label_column]} | '
        f'ROI {worst_row["ROI %"]}% | '
        f'Profit ${worst_row["Profit_Loss"]:.2f}'
    )

    return best_text, worst_text

def value_rating(edge_percent, roi_percent):
    if edge_percent >= 8 and roi_percent >= 8:
        return "Strong Value"

    if edge_percent >= 4 and roi_percent >= 4:
        return "Value"

    if edge_percent >= 1 and roi_percent >= 1:
        return "Small Edge"

    return "No Value"


def multi_eligible(value_status, risk):
    if value_status in ["Strong Value", "Value"] and risk != "HIGH":
        return "YES"

    return "NO"

def find_best_odds_for_tip(odds_data, match_name, tipped_team):
    possible_names = TEAM_NAME_MAP.get(
        tipped_team,
        [tipped_team]
    )

    if isinstance(possible_names, str):
        possible_names = [possible_names]

    best_price = None
    best_bookmaker = None
    matched_name = None

    for game in odds_data:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        game_teams = [home_team, away_team]

        matched_team_name = None

        for name in possible_names:
            if name in game_teams:
                matched_team_name = name
                break

        if matched_team_name is None:
            continue

        for bookmaker in game.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == matched_team_name:
                        price = outcome.get("price")

                        if best_price is None or price > best_price:
                            best_price = price
                            best_bookmaker = bookmaker_name
                            matched_name = matched_team_name

    if best_price is None:
        return None, None

    return best_price, best_bookmaker
    

# ==========================
# PREDICTIONS
# ==========================

def format_margin_tip(home, away, predicted_margin):
    margin_value = abs(predicted_margin)
    rounded_margin = round(margin_value)

    if predicted_margin >= 0:
        team = home
    else:
        team = away

    if rounded_margin == 0:
        return f"{team} by less than 1"

    return f"{team} by {rounded_margin}"


def make_predictions(round_number):
    games_data = squiggle(
        "games",
        year=YEAR
    )

    tips_data = squiggle(
        "tips",
        year=YEAR
    )

    games_df = pd.DataFrame(
        games_data["games"]
    )

    tips_df = pd.DataFrame(
        tips_data["tips"]
    )

    elo = build_elo(games_df)

    upcoming = games_df[
        (games_df["round"] == round_number)
        & (games_df["complete"] < 100)
    ].copy()

    rows = []

    for _, game in upcoming.iterrows():
        home = game["hteam"]
        away = game["ateam"]
        game_id = game["id"]

        home_rating = elo[home] + HOME_ADVANTAGE
        away_rating = elo[away]

        home_prob = expected_score(
            home_rating,
            away_rating
        )

        rating_difference = home_rating - away_rating
        predicted_margin = rating_difference / MARGIN_DIVISOR

        margin_tip = format_margin_tip(
            home,
            away,
            predicted_margin
        )
        
        abs_margin = abs(predicted_margin)

        if abs_margin < 10:
            margin_category = "Close Game"
        elif abs_margin < 25:
            margin_category = "Solid Margin"
        else:
            margin_category = "Strong Margin"

        away_prob = 1 - home_prob

        if home_prob >= away_prob:
            elo_tip = home
        else:
            elo_tip = away

        confidence = max(
            home_prob,
            away_prob
        ) * 100

        squiggle_tip, squiggle_votes = (
            get_squiggle_consensus(
                tips_df,
                game_id,
                home,
                away
            )
        )

        risk = risk_rating(
            elo_tip,
            squiggle_tip,
            confidence
        )

        edge_score = calculate_edge_score(
            confidence,
            risk
        )

        rows.append({
            "Match": f"{home} v {away}",
            "Final Tip": elo_tip,
            "Predicted Margin": margin_tip,
            "Raw Margin": round(predicted_margin, 1),
            "Margin Category": margin_category,
            "Elo Confidence": round(confidence, 1),
            "Edge Score": edge_score,
            "Home Win %": round(home_prob * 100, 1),
            "Away Win %": round(away_prob * 100, 1),
            "Squiggle Tip": squiggle_tip,
            "Squiggle Votes": squiggle_votes,
            "Risk": risk
        })

    return pd.DataFrame(rows)


def run_historical_backtest(start_year, end_year, stake):
    rows = []

    ratings = {}

    for season in range(start_year, end_year + 1):

        games_data = squiggle(
            "games",
            year=season
        )

        games_df = pd.DataFrame(
            games_data["games"]
        )

        completed_games = games_df[
            games_df["complete"] == 100
        ].copy()

        completed_games = completed_games.sort_values(
            ["date"]
        )

        season_teams = sorted(
            set(completed_games["hteam"].dropna())
            | set(completed_games["ateam"].dropna())
        )

        for team in season_teams:
            if team not in ratings:
                ratings[team] = START_RATING

        for _, game in completed_games.iterrows():
            home = game["hteam"]
            away = game["ateam"]

            if home not in ratings:
                ratings[home] = START_RATING

            if away not in ratings:
                ratings[away] = START_RATING

            home_rating = ratings[home] + HOME_ADVANTAGE
            away_rating = ratings[away]

            home_prob = expected_score(
                home_rating,
                away_rating
            )

            away_prob = 1 - home_prob

            if home_prob >= away_prob:
                tip = home
                confidence = home_prob * 100
            else:
                tip = away
                confidence = away_prob * 100

            if game["hscore"] > game["ascore"]:
                winner = home
                home_actual = 1
            elif game["hscore"] < game["ascore"]:
                winner = away
                home_actual = 0
            else:
                winner = "DRAW"
                home_actual = 0.5

            correct = tip == winner

            if winner == "DRAW":
                profit_loss = 0
            elif correct:
                profit_loss = stake * 0.90
            else:
                profit_loss = -stake

            margin = abs(
                game["hscore"] - game["ascore"]
            )

            margin_multiplier = max(
                1,
                margin / 30
            )

            home_expected = expected_score(
                home_rating,
                away_rating
            )

            change = (
                K_FACTOR
                * margin_multiplier
                * (home_actual - home_expected)
            )

            ratings[home] += change
            ratings[away] -= change

            rows.append({
                "Season": season,
                "Round": game.get("round"),
                "Date": game.get("date"),
                "Match": f"{home} v {away}",
                "Tip": tip,
                "Winner": winner,
                "Correct": correct,
                "Confidence": round(confidence, 1),
                "Risk": backtest_risk_rating(confidence),
                "Stake": stake,
                "Profit/Loss": round(profit_loss, 2)
            })

    return pd.DataFrame(rows)
# ==========================
# HERO SECTION
# ==========================

st.markdown(
    "<h1 style='text-align:center; font-size:82px; color:white; margin-bottom:0;'>SBT EDGE</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align:center; color:#00A3FF; margin-top:10px;'>AFL Prediction Engine</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#CCCCCC; font-size:18px;'>Validated Elo Model + Margin Predictor + Odds/EV Layer</p>",
    unsafe_allow_html=True
)

# ==========================
# KPI CARDS
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model Accuracy",
        "69.36%"
    )

with col2:
    st.metric(
        "Avg Margin Error",
        "28.63 pts"
    )

with col3:
    st.metric(
        "Historical Games",
        "2,716"
    )

with col4:
    st.metric(
        "Model Version",
        "V5.4 Risk Rule Finder"
    )

st.info(
    "V3 adds bookmaker odds comparison using manual input or auto-fetched AFL odds. "
    "The app calculates break-even probability, model edge, expected ROI, value rating, "
    "and best available bookmaker price where odds data is available."
)
# ==========================
# ROUND SELECTOR
# ==========================

round_number = st.number_input(
    "Round to predict",
    min_value=1,
    max_value=30,
    value=13,
    step=1
)

try:
    odds_api_key = st.secrets["ODDS_API_KEY"]
except Exception:
    odds_api_key = None
    st.warning(
        "Odds API key is not configured. "
        "Auto odds will not load until ODDS_API_KEY is added to Streamlit Secrets."
    )

if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None

if "has_run_predictor" not in st.session_state:
    st.session_state.has_run_predictor = False

if "backtest_df" not in st.session_state:
    st.session_state.backtest_df = None

if "has_run_backtest" not in st.session_state:
    st.session_state.has_run_backtest = False

# ==========================
# RESULTS / ROI TRACKER
# ==========================

st.subheader("📊 Results + ROI Tracker")

bankroll_col1, bankroll_col2 = st.columns(2)

with bankroll_col1:
    starting_bankroll = st.number_input(
        "Starting Bankroll",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

with bankroll_col2:
    unit_size = st.number_input(
        "Unit Size",
        min_value=1.0,
        value=10.0,
        step=1.0
    )

uploaded_tracker = st.file_uploader(
    "Upload Bet Tracker CSV",
    type=["csv"]
)

if uploaded_tracker is not None:

    tracker_df = pd.read_csv(uploaded_tracker)

    # Clean editable text columns so Streamlit data_editor accepts them
    for text_col in ["Result", "Notes"]:
        if text_col not in tracker_df.columns:
            tracker_df[text_col] = ""

        tracker_df[text_col] = tracker_df[text_col].fillna("").astype(str)

    st.success(
        f"Loaded {len(tracker_df)} tracked bets."
    )

    st.caption(
        "Edit Stake, Result, Profit/Loss, and Notes directly here. "
        "Valid results: WIN, LOSS, PUSH, VOID, DRAW."
    )

    editable_tracker_df = st.data_editor(
        tracker_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Result": st.column_config.SelectboxColumn(
                "Result",
                options=["", "WIN", "LOSS", "PUSH", "VOID", "DRAW"],
                help="Select the final result"
            ),
            "Stake": st.column_config.NumberColumn(
                "Stake",
                min_value=0.0,
                step=1.0,
                format="$%.2f"
            ),
            "Profit/Loss": st.column_config.NumberColumn(
                "Profit/Loss",
                step=1.0,
                format="$%.2f",
                help="Optional. Leave blank to auto-calculate."
            ),
            "Notes": st.column_config.TextColumn(
                "Notes"
            )
        },
        disabled=[
            "Round",
            "Match",
            "Final Tip",
            "Bet Type",
            "Predicted Margin",
            "Bookmaker Odds",
            "Best Bookmaker",
            "Elo Confidence",
            "Break Even %",
            "Model Edge %",
            "Expected ROI %",
            "Value Rating",
            "Multi Eligible",
            "Risk",
            "Recommended Units",
            "Recommended Stake"
        ]
    )

    tracker_df = editable_tracker_df

    st.download_button(
        label="Download Updated Tracker CSV",
        data=tracker_df.to_csv(index=False),
        file_name="updated_bet_tracker.csv",
        mime="text/csv"
    )
    
    tracker_df["Clean Result"] = tracker_df["Result"].apply(
        clean_result
    )

    invalid_results = tracker_df[
        tracker_df["Result"].notna()
        & (tracker_df["Result"].astype(str).str.strip() != "")
        & (tracker_df["Clean Result"] == "")
    ].copy()

    if not invalid_results.empty:
        st.warning(
            f"Ignored {len(invalid_results)} invalid result entries. "
            "Use only WIN, LOSS, PUSH, VOID, or DRAW."
        )

    settled_df = tracker_df[
        tracker_df["Clean Result"] != ""
    ].copy()

    settled_df["Result"] = settled_df["Clean Result"]

    if settled_df.empty:
        st.info(
            "No settled bets yet. Fill in Result and Profit/Loss after games are completed."
        )

    else:
        settled_df["Stake"] = pd.to_numeric(
            settled_df["Stake"],
            errors="coerce"
        )

        if "Recommended Stake" in settled_df.columns:
            settled_df["Recommended Stake"] = pd.to_numeric(
                settled_df["Recommended Stake"],
                errors="coerce"
            ).fillna(0)

            settled_df["Stake"] = settled_df["Stake"].fillna(
                settled_df["Recommended Stake"]
            )

        settled_df["Stake"] = settled_df["Stake"].fillna(0)

        settled_df["Bookmaker Odds"] = pd.to_numeric(
            settled_df["Bookmaker Odds"],
            errors="coerce"
        ).fillna(0)

        if "Profit/Loss" not in settled_df.columns:
            settled_df["Profit/Loss"] = ""

        settled_df["Profit/Loss"] = pd.to_numeric(
            settled_df["Profit/Loss"],
            errors="coerce"
        )

        settled_df["Auto Profit/Loss"] = settled_df.apply(
            lambda row: calculate_profit_loss(
                row["Stake"],
                row["Result"],
                row["Bookmaker Odds"]
            ),
            axis=1
        )

        settled_df["Final Profit/Loss"] = settled_df["Profit/Loss"].fillna(
            settled_df["Auto Profit/Loss"]
        )

        total_bets = len(settled_df)
        total_staked = settled_df["Stake"].sum()
        total_profit = settled_df["Final Profit/Loss"].sum()

        settled_df = settled_df.reset_index(drop=True)

        settled_df["Bet Number"] = settled_df.index + 1

        settled_df["Running Profit/Loss"] = settled_df[
            "Final Profit/Loss"
        ].cumsum()

        settled_df["Running Bankroll"] = (
            starting_bankroll
            + settled_df["Running Profit/Loss"]
        )

        if unit_size > 0:
            total_staked_units = total_staked / unit_size
            profit_units = total_profit / unit_size
        else:
            total_staked_units = 0
            profit_units = 0

        ending_bankroll = starting_bankroll + total_profit

        if starting_bankroll > 0:
            bankroll_growth = (total_profit / starting_bankroll) * 100
        else:
            bankroll_growth = 0

        if total_staked > 0:
            roi_percent = (total_profit / total_staked) * 100
        else:
            roi_percent = 0

        wins = settled_df[
            settled_df["Result"] == "WIN"
        ]

        win_rate = (len(wins) / total_bets) * 100

        roi_col1, roi_col2, roi_col3, roi_col4, roi_col5, roi_col6 = st.columns(6)

        with roi_col1:
            st.metric("Settled Bets", total_bets)

        with roi_col2:
            st.metric("Total Staked", f"${total_staked:.2f}")

        with roi_col3:
            st.metric("Profit / Loss", f"${total_profit:.2f}")

        with roi_col4:
            st.metric("ROI", f"{roi_percent:.1f}%")

        with roi_col5:
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with roi_col6:
            st.metric("Ending Bankroll", f"${ending_bankroll:.2f}")

        bank_col1, bank_col2, bank_col3 = st.columns(3)

        with bank_col1:
            st.metric(
                "Total Staked Units",
                f"{total_staked_units:.1f}u"
        )

        with bank_col2:
            st.metric(
                "Profit / Loss Units",
                f"{profit_units:.1f}u"
        )

        with bank_col3:
            st.metric(
                "Bankroll Growth",
                f"{bankroll_growth:.1f}%"
        )

        st.subheader("📈 Bankroll Performance Chart")

        chart_df = settled_df[
            [
                "Bet Number",
                "Running Bankroll"
            ]
        ].copy()

        st.line_chart(
            chart_df,
            x="Bet Number",
            y="Running Bankroll",
            height=320
        )

        st.subheader("Settled Bet Details")

        settled_display = settled_df[
            [
                "Bet Number",
                "Match",
                "Final Tip",
                "Bookmaker Odds",
                "Stake",
                "Result",
                "Auto Profit/Loss",
                "Final Profit/Loss",
                "Running Bankroll",
                "Value Rating",
                "Risk"
            ]
        ].copy()

        st.dataframe(
            settled_display,
            width="stretch",
            hide_index=True
        )

        st.subheader("Performance by Value Rating")

        value_summary = settled_df.groupby(
            "Value Rating"
        ).agg(
            Bets=("Match", "count"),
            Total_Stake=("Stake", "sum"),
            Profit_Loss=("Final Profit/Loss", "sum")
        ).reset_index()

        value_summary["ROI %"] = (
            value_summary["Profit_Loss"]
            / value_summary["Total_Stake"]
            * 100
        ).fillna(0).round(1)

        st.dataframe(
            value_summary,
            width="stretch",
            hide_index=True
        )

        st.subheader("Performance by Risk Level")

        risk_summary = settled_df.groupby(
            "Risk"
        ).agg(
            Bets=("Match", "count"),
            Total_Stake=("Stake", "sum"),
            Profit_Loss=("Final Profit/Loss", "sum")
        ).reset_index()

        risk_summary["ROI %"] = (
            risk_summary["Profit_Loss"]
            / risk_summary["Total_Stake"]
            * 100
        ).fillna(0).round(1)

        st.dataframe(
            risk_summary,
            width="stretch",
            hide_index=True
        )

        st.subheader("Performance by Round")

        round_summary = settled_df.copy()

        round_summary["Win Flag"] = (
            round_summary["Result"] == "WIN"
        ).astype(int)

        round_summary = round_summary.groupby(
            "Round"
        ).agg(
            Bets=("Match", "count"),
            Total_Stake=("Stake", "sum"),
            Profit_Loss=("Final Profit/Loss", "sum"),
            Wins=("Win Flag", "sum")
        ).reset_index()

        round_summary["ROI %"] = (
            round_summary["Profit_Loss"]
            / round_summary["Total_Stake"]
            * 100
        ).fillna(0).round(1)

        round_summary["Win Rate %"] = (
            round_summary["Wins"]
            / round_summary["Bets"]
            * 100
        ).fillna(0).round(1)

        st.dataframe(
            round_summary,
            width="stretch",
            hide_index=True
        )

        st.subheader("Profit/Loss by Round")

        round_chart = round_summary[
            [
                "Round",
                "Profit_Loss"
            ]
        ].copy()

        st.bar_chart(
            round_chart,
            x="Round",
            y="Profit_Loss",
            height=320
        )
        
        st.subheader("Performance by Bet Type")

        bet_type_summary = settled_df.copy()

        bet_type_summary["Win Flag"] = (
            bet_type_summary["Result"] == "WIN"
        ).astype(int)

        bet_type_summary = bet_type_summary.groupby(
            "Bet Type"
        ).agg(
            Bets=("Match", "count"),
            Total_Stake=("Stake", "sum"),
            Profit_Loss=("Final Profit/Loss", "sum"),
            Wins=("Win Flag", "sum")
        ).reset_index()

        bet_type_summary["ROI %"] = (
            bet_type_summary["Profit_Loss"]
            / bet_type_summary["Total_Stake"]
            * 100
        ).fillna(0).round(1)

        bet_type_summary["Win Rate %"] = (
            bet_type_summary["Wins"]
            / bet_type_summary["Bets"]
            * 100
        ).fillna(0).round(1)

        st.dataframe(
            bet_type_summary,
            width="stretch",
            hide_index=True
        )
        st.subheader("Profit/Loss by Bet Type")

        bet_type_chart = bet_type_summary[
            [
                "Bet Type",
                "Profit_Loss"
            ]
        ].copy()

        st.bar_chart(
            bet_type_chart,
            x="Bet Type",
            y="Profit_Loss",
            height=320
        )        

        # --------------------------
        # BEST / WORST PERFORMANCE SUMMARY
        # --------------------------

        st.subheader("🏆 Best / Worst Performance Summary")

        best_value, worst_value = best_worst_summary(
            value_summary,
            "Value Rating"
        )

        best_risk, worst_risk = best_worst_summary(
            risk_summary,
            "Risk"
        )

        best_round, worst_round = best_worst_summary(
            round_summary,
            "Round"
        )

        best_bet_type, worst_bet_type = best_worst_summary(
            bet_type_summary,
            "Bet Type"
        )

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.success(
                f"Best Value Rating: {best_value}"
            )

            st.success(
                f"Best Risk Level: {best_risk}"
            )

            st.success(
                f"Best Round: {best_round}"
            )

            st.success(
                f"Best Bet Type: {best_bet_type}"
            )

        with summary_col2:
            st.warning(
                f"Worst Value Rating: {worst_value}"
            )

            st.warning(
                f"Worst Risk Level: {worst_risk}"
            )

            st.warning(
                f"Worst Round: {worst_round}"
            )

            st.warning(
                f"Worst Bet Type: {worst_bet_type}"
            )
# ==========================
# V5 HISTORICAL BACKTEST
# ==========================

st.subheader("🧪 V5 Historical Backtest")

st.caption(
    "Backtests the SBT EDGE Elo model across completed historical AFL games. "
    "V5.0 uses flat staking without historical bookmaker odds. "
    "True EV/value backtesting comes next when historical odds are added."
)

backtest_col1, backtest_col2, backtest_col3 = st.columns(3)

with backtest_col1:
    backtest_start_year = st.number_input(
        "Backtest Start Year",
        min_value=2012,
        max_value=YEAR,
        value=2012,
        step=1
    )

with backtest_col2:
    backtest_end_year = st.number_input(
        "Backtest End Year",
        min_value=2012,
        max_value=YEAR,
        value=YEAR,
        step=1
    )

with backtest_col3:
    backtest_stake = st.number_input(
        "Backtest Flat Stake",
        min_value=1.0,
        value=10.0,
        step=1.0
    )

if st.button("Run Historical Backtest"):

    if backtest_start_year > backtest_end_year:
        st.error("Start year must be before or equal to end year.")

    else:
        with st.spinner("Running historical backtest..."):
            backtest_df = run_historical_backtest(
                backtest_start_year,
                backtest_end_year,
                backtest_stake
            )

        st.session_state.backtest_df = backtest_df
        st.session_state.has_run_backtest = True


if st.session_state.has_run_backtest:

    backtest_df = st.session_state.backtest_df

    if backtest_df is None or backtest_df.empty:
        st.warning("No completed historical games found.")

    else:
        total_backtest_bets = len(backtest_df)

        correct_bets = backtest_df[
            backtest_df["Correct"] == True
        ]

        accuracy = (
            len(correct_bets)
            / total_backtest_bets
            * 100
        )

        total_backtest_staked = backtest_df["Stake"].sum()
        total_backtest_profit = backtest_df["Profit/Loss"].sum()

        if total_backtest_staked > 0:
            backtest_roi = (
                total_backtest_profit
                / total_backtest_staked
                * 100
            )
        else:
            backtest_roi = 0

        backtest_df = backtest_df.reset_index(drop=True)

        backtest_df["Bet Number"] = backtest_df.index + 1

        backtest_df["Running Profit/Loss"] = backtest_df[
            "Profit/Loss"
        ].cumsum()

        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)

        with bt_col1:
            st.metric(
                "Backtest Bets",
                total_backtest_bets
            )

        with bt_col2:
            st.metric(
                "Tip Accuracy",
                f"{accuracy:.1f}%"
            )

        with bt_col3:
            st.metric(
                "Profit / Loss",
                f"${total_backtest_profit:.2f}"
            )

        with bt_col4:
            st.metric(
                "Simulated ROI",
                f"{backtest_roi:.1f}%"
            )

        st.subheader("Backtest Running Profit/Loss")

        st.line_chart(
            backtest_df[
                [
                    "Bet Number",
                    "Running Profit/Loss"
                ]
            ],
            x="Bet Number",
            y="Running Profit/Loss",
            height=320
        )

        st.subheader("Backtest by Season")

        season_summary = backtest_df.groupby(
            "Season"
        ).agg(
            Bets=("Match", "count"),
            Wins=("Correct", "sum"),
            Total_Stake=("Stake", "sum"),
            Profit_Loss=("Profit/Loss", "sum")
        ).reset_index()

        season_summary["Accuracy %"] = (
            season_summary["Wins"]
            / season_summary["Bets"]
            * 100
        ).round(1)

        season_summary["ROI %"] = (
            season_summary["Profit_Loss"]
            / season_summary["Total_Stake"]
            * 100
        ).round(1)

        st.dataframe(
            season_summary,
            width="stretch",
            hide_index=True
        )

        st.subheader("Backtest by Confidence Threshold")

        confidence_thresholds = [
            55,
            60,
            65,
            70,
            75
        ]

        confidence_rows = []

        for threshold in confidence_thresholds:
            filtered_df = backtest_df[
                backtest_df["Confidence"] >= threshold
            ].copy()

            if filtered_df.empty:
                confidence_rows.append({
                    "Confidence Threshold": f"{threshold}%+",
                    "Bets": 0,
                    "Wins": 0,
                    "Accuracy %": 0,
                    "Total Stake": 0,
                    "Profit/Loss": 0,
                    "ROI %": 0
                })
                continue

            bets = len(filtered_df)
            wins = len(
                filtered_df[
                    filtered_df["Correct"] == True
                ]
            )

            total_stake = filtered_df["Stake"].sum()
            profit_loss = filtered_df["Profit/Loss"].sum()

            accuracy_pct = (
                wins
                / bets
                * 100
            )

            if total_stake > 0:
                roi_pct = (
                    profit_loss
                    / total_stake
                    * 100
                )
            else:
                roi_pct = 0

            confidence_rows.append({
                "Confidence Threshold": f"{threshold}%+",
                "Bets": bets,
                "Wins": wins,
                "Accuracy %": round(accuracy_pct, 1),
                "Total Stake": round(total_stake, 2),
                "Profit/Loss": round(profit_loss, 2),
                "ROI %": round(roi_pct, 1)
                })

        confidence_summary = pd.DataFrame(
            confidence_rows
        )

        st.dataframe(
            confidence_summary,
            width="stretch",
            hide_index=True
        )

# --------------------------
# BEST CONFIDENCE RULE FINDER
# --------------------------

        st.subheader("🏆 Best Confidence Rule Finder")

        if not confidence_summary.empty:

            best_roi_rule = confidence_summary.sort_values(
                "ROI %",
                ascending=False
            ).iloc[0]

            best_profit_rule = confidence_summary.sort_values(
                "Profit/Loss",
                ascending=False
            ).iloc[0]

            best_accuracy_rule = confidence_summary.sort_values(
                "Accuracy %",
                ascending=False
            ).iloc[0]

            balanced_summary = confidence_summary.copy()

            balanced_summary["Balance Score"] = (
                balanced_summary["Accuracy %"]
                + balanced_summary["ROI %"]
            )

            best_balanced_rule = balanced_summary.sort_values(
                "Balance Score",
                ascending=False
            ).iloc[0]

            best_col1, best_col2, best_col3, best_col4 = st.columns(4)

            with best_col1:
                st.metric(
                    "Best ROI Rule",
                    best_roi_rule["Confidence Threshold"],
                    f'{best_roi_rule["ROI %"]}% ROI'
                )

            with best_col2:
                st.metric(
                    "Best Profit Rule",
                    best_profit_rule["Confidence Threshold"],
                    f'${best_profit_rule["Profit/Loss"]:.2f}'
                )

            with best_col3:
                st.metric(
                    "Best Accuracy Rule",
                    best_accuracy_rule["Confidence Threshold"],
                    f'{best_accuracy_rule["Accuracy %"]}%'
                )

            with best_col4:
                st.metric(
                    "Best Balanced Rule",
                    best_balanced_rule["Confidence Threshold"],
                    f'{best_balanced_rule["Balance Score"]:.1f} score'
                )

            st.info(
                f'SBT EDGE historical sweet spot: '
                f'{best_balanced_rule["Confidence Threshold"]} confidence '
                f'with {best_balanced_rule["Accuracy %"]}% accuracy '
                f'and {best_balanced_rule["ROI %"]}% simulated ROI.'
            )

        st.subheader("ROI by Confidence Threshold")

        st.bar_chart(
            confidence_summary[
                [
                    "Confidence Threshold",
                    "ROI %"
                ]
            ],
            x="Confidence Threshold",
            y="ROI %",
            height=320
        )
        
        # --------------------------
        # RISK + CONFIDENCE RULE FINDER
        # --------------------------

        st.subheader("🧠 Risk + Confidence Rule Finder")

        rule_sets = [
            {
                "Rule": "55%+ Any Risk",
                "Min Confidence": 55,
                "Allowed Risk": ["LOW", "MEDIUM", "HIGH"]
            },
            {
                "Rule": "60%+ Any Risk",
                "Min Confidence": 60,
                "Allowed Risk": ["LOW", "MEDIUM", "HIGH"]
            },
            {
                "Rule": "70%+ Any Risk",
                "Min Confidence": 70,
                "Allowed Risk": ["LOW", "MEDIUM", "HIGH"]
            },
            {
                "Rule": "75%+ Any Risk",
                "Min Confidence": 75,
                "Allowed Risk": ["LOW", "MEDIUM", "HIGH"]
            },
            {
                "Rule": "60%+ LOW/MEDIUM Risk",
                "Min Confidence": 60,
                "Allowed Risk": ["LOW", "MEDIUM"]
            },
            {
                "Rule": "70%+ LOW/MEDIUM Risk",
                "Min Confidence": 70,
                "Allowed Risk": ["LOW", "MEDIUM"]
            },
            {
                "Rule": "75%+ LOW Risk",
                "Min Confidence": 75,
                "Allowed Risk": ["LOW"]
            }
        ]

        rule_rows = []

        for rule in rule_sets:
            rule_df = backtest_df[
                (backtest_df["Confidence"] >= rule["Min Confidence"])
                & (backtest_df["Risk"].isin(rule["Allowed Risk"]))
            ].copy()

            if rule_df.empty:
                rule_rows.append({
                    "Rule": rule["Rule"],
                    "Bets": 0,
                    "Wins": 0,
                    "Accuracy %": 0,
                    "Total Stake": 0,
                    "Profit/Loss": 0,
                    "ROI %": 0
                })
                continue

            rule_bets = len(rule_df)

            rule_wins = len(
                rule_df[
                    rule_df["Correct"] == True
                ]
            )

            rule_accuracy = (
                rule_wins
                / rule_bets
                * 100
            )

            rule_total_stake = rule_df["Stake"].sum()
            rule_profit = rule_df["Profit/Loss"].sum()

            if rule_total_stake > 0:
                rule_roi = (
                    rule_profit
                    / rule_total_stake
                    * 100
                )
            else:
                rule_roi = 0

            rule_rows.append({
                "Rule": rule["Rule"],
                "Bets": rule_bets,
                "Wins": rule_wins,
                "Accuracy %": round(rule_accuracy, 1),
                "Total Stake": round(rule_total_stake, 2),
                "Profit/Loss": round(rule_profit, 2),
                "ROI %": round(rule_roi, 1)
            })

        rule_summary = pd.DataFrame(rule_rows)

        st.dataframe(
            rule_summary,
            width="stretch",
            hide_index=True
        )

        best_risk_roi_rule = rule_summary.sort_values(
            "ROI %",
            ascending=False
        ).iloc[0]

        best_risk_profit_rule = rule_summary.sort_values(
            "Profit/Loss",
            ascending=False
        ).iloc[0]

        risk_balanced_summary = rule_summary.copy()

        risk_balanced_summary["Balance Score"] = (
            risk_balanced_summary["Accuracy %"]
            + risk_balanced_summary["ROI %"]
        )

        best_risk_balanced_rule = risk_balanced_summary.sort_values(
            "Balance Score",
            ascending=False
        ).iloc[0]

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            st.metric(
                "Best Risk ROI Rule",
                best_risk_roi_rule["Rule"],
                f'{best_risk_roi_rule["ROI %"]}% ROI'
            )

        with risk_col2:
            st.metric(
                "Best Risk Profit Rule",
                best_risk_profit_rule["Rule"],
                f'${best_risk_profit_rule["Profit/Loss"]:.2f}'
            )

        with risk_col3:
            st.metric(
                "Best Risk Balanced Rule",
                best_risk_balanced_rule["Rule"],
                f'{best_risk_balanced_rule["Balance Score"]:.1f} score'
            )

        st.subheader("ROI by Risk + Confidence Rule")

        st.bar_chart(
            rule_summary[
                [
                    "Rule",
                    "ROI %"
                ]
            ],
            x="Rule",
            y="ROI %",
            height=360
        )

        st.info(
            f'SBT EDGE combined rule sweet spot: '
            f'{best_risk_balanced_rule["Rule"]} '
            f'with {best_risk_balanced_rule["Accuracy %"]}% accuracy '
            f'and {best_risk_balanced_rule["ROI %"]}% simulated ROI.'
        )
        # --------------------------
        # MINIMUM CONFIDENCE BETTING RULE
        # --------------------------

        st.subheader("🎯 Minimum Confidence Betting Rule")

        selected_min_confidence = st.selectbox(
            "Only bet when confidence is at least",
            options=[55, 60, 65, 70, 75],
            index=3
        )

        filtered_backtest_df = backtest_df[
            backtest_df["Confidence"] >= selected_min_confidence
        ].copy()

        if filtered_backtest_df.empty:
            st.warning(
                "No bets found for this confidence rule."
            )

        else:
            filtered_bets = len(filtered_backtest_df)

            filtered_wins = len(
                filtered_backtest_df[
                    filtered_backtest_df["Correct"] == True
                ]
            )

            filtered_accuracy = (
                filtered_wins
                / filtered_bets
                * 100
            )

            filtered_total_staked = filtered_backtest_df["Stake"].sum()
            filtered_profit = filtered_backtest_df["Profit/Loss"].sum()

            if filtered_total_staked > 0:
                filtered_roi = (
                    filtered_profit
                    / filtered_total_staked
                    * 100
                )
            else:
                filtered_roi = 0

            rule_col1, rule_col2, rule_col3, rule_col4 = st.columns(4)

            with rule_col1:
                st.metric(
                    "Filtered Bets",
                    filtered_bets
                )

            with rule_col2:
                st.metric(
                    "Filtered Accuracy",
                    f"{filtered_accuracy:.1f}%"
                )

            with rule_col3:
                st.metric(
                    "Filtered Profit / Loss",
                    f"${filtered_profit:.2f}"
                )

            with rule_col4:
                st.metric(
                    "Filtered ROI",
                    f"{filtered_roi:.1f}%"
                )

            filtered_backtest_df = filtered_backtest_df.reset_index(
                drop=True
            )

            filtered_backtest_df["Filtered Bet Number"] = (
                filtered_backtest_df.index + 1
            )

            filtered_backtest_df["Filtered Running Profit/Loss"] = (
                filtered_backtest_df["Profit/Loss"].cumsum()
            )

            st.subheader("Filtered Running Profit/Loss")

            st.line_chart(
                filtered_backtest_df[
                    [
                        "Filtered Bet Number",
                        "Filtered Running Profit/Loss"
                    ]
                ],
                x="Filtered Bet Number",
                y="Filtered Running Profit/Loss",
                height=320
            )

            st.subheader("Filtered Backtest by Season")

            filtered_season_summary = filtered_backtest_df.groupby(
                "Season"
            ).agg(
                Bets=("Match", "count"),
                Wins=("Correct", "sum"),
                Total_Stake=("Stake", "sum"),
                Profit_Loss=("Profit/Loss", "sum")
            ).reset_index()

            filtered_season_summary["Accuracy %"] = (
                filtered_season_summary["Wins"]
                / filtered_season_summary["Bets"]
                * 100
            ).round(1)

            filtered_season_summary["ROI %"] = (
                filtered_season_summary["Profit_Loss"]
                / filtered_season_summary["Total_Stake"]
                * 100
            ).round(1)

            st.dataframe(
                filtered_season_summary,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                label="Download Filtered Backtest CSV",
                data=filtered_backtest_df.to_csv(index=False),
                file_name=f"sbt_edge_v5_filtered_{selected_min_confidence}_confidence.csv",
                mime="text/csv"
            )

        st.download_button(
            label="Download Backtest CSV",
            data=backtest_df.to_csv(index=False),
            file_name="sbt_edge_v5_historical_backtest.csv",
            mime="text/csv"
        )


# ==========================
# RUN PREDICTOR
# ==========================

if st.button("Run Predictor"):

    with st.spinner("Running SBT EDGE..."):
        df = make_predictions(round_number)

    st.session_state.predictions_df = df
    st.session_state.has_run_predictor = True


if st.session_state.has_run_predictor:

    df = st.session_state.predictions_df

    if df is None or df.empty:
        st.warning("No upcoming games found for this round.")

    else:
        st.subheader(f"🏉 Round {round_number} Predictions")

        # --------------------------
        # ODDS INPUT
        # --------------------------

        st.subheader("💰 Bookmaker Odds Input")

        st.caption(
            "Enter decimal odds for the team SBT EDGE has tipped. "
            "Example: 1.57 means $1.57 return for every $1 staked."
        )

        auto_odds_enabled = False
        odds_data = []

        if odds_api_key:
            try:
                odds_data = fetch_afl_odds(odds_api_key)
                auto_odds_enabled = True
                st.success(f"Auto odds loaded. Found {len(odds_data)} AFL games.")
            except Exception as e:
                st.warning(f"Could not fetch auto odds: {e}")

        odds_values = []
        bookmaker_values = []

        for index, row in df.iterrows():

            default_odds = 1.50
            best_bookmaker = "Manual / No API Match"

            if auto_odds_enabled:
                best_odds, api_bookmaker = find_best_odds_for_tip(
                    odds_data,
                    row["Match"],
                    row["Final Tip"]
                )

                if best_odds is not None:
                    default_odds = float(best_odds)
                    best_bookmaker = api_bookmaker

            odds = st.number_input(
                f'Odds for {row["Final Tip"]} — {row["Match"]}',
                min_value=1.01,
                max_value=20.00,
                value=default_odds,
                step=0.01,
                key=f"odds_{index}"
            )

            odds_values.append(odds)
            bookmaker_values.append(best_bookmaker)

        df["Bookmaker Odds"] = odds_values
        df["Best Bookmaker"] = bookmaker_values

        # --------------------------
        # ODDS / EV CALCULATION
        # --------------------------

        break_even_list = []
        edge_list = []
        roi_list = []
        value_list = []
        multi_list = []

        for _, row in df.iterrows():
            model_probability = row["Elo Confidence"] / 100
            decimal_odds = row["Bookmaker Odds"]

            break_even = break_even_probability(decimal_odds)
            roi = expected_roi(model_probability, decimal_odds)

            break_even_percent = break_even * 100
            roi_percent = roi * 100
            edge_percent = row["Elo Confidence"] - break_even_percent

            status = value_rating(
                edge_percent,
                roi_percent
            )

            multi_status = multi_eligible(
                status,
                row["Risk"]
            )

            break_even_list.append(round(break_even_percent, 1))
            edge_list.append(round(edge_percent, 1))
            roi_list.append(round(roi_percent, 1))
            value_list.append(status)
            multi_list.append(multi_status)

        df["Break Even %"] = break_even_list
        df["Model Edge %"] = edge_list
        df["Expected ROI %"] = roi_list
        df["Value Rating"] = value_list
        df["Multi Eligible"] = multi_list

        # --------------------------
        # BEST PICK OF THE ROUND
        # --------------------------

        best_pick = df.sort_values(
            "Edge Score",
            ascending=False
        ).iloc[0]

        st.success(
            f'🏆 BEST PICK OF THE ROUND: '
            f'{best_pick["Predicted Margin"]} '
            f'({best_pick["Match"]}) | '
            f'{best_pick["Margin Category"]} | '
            f'Confidence {best_pick["Elo Confidence"]}% | '
            f'Edge {best_pick["Edge Score"]}/10'
        )

        # --------------------------
        # BEST VALUE PICK
        # --------------------------

        value_candidates = df[
            df["Value Rating"].isin(
                ["Strong Value", "Value", "Small Edge"]
            )
        ].copy()

        if not value_candidates.empty:
            best_value = value_candidates.sort_values(
                "Expected ROI %",
                ascending=False
            ).iloc[0]

            st.success(
                f'💰 BEST VALUE PICK: '
                f'{best_value["Predicted Margin"]} '
                f'@ ${best_value["Bookmaker Odds"]:.2f} | '
                f'{best_value["Best Bookmaker"]} | '
                f'ROI {best_value["Expected ROI %"]}% | '
                f'Edge {best_value["Model Edge %"]}% | '
                f'{best_value["Value Rating"]}'
            )
        else:
            st.warning("No value picks found from the entered odds.")

        # --------------------------
        # VALUE BETS TABLE
        # --------------------------

        st.subheader("💎 Value Bets Table")

        value_table = df[
            df["Value Rating"].isin(
                ["Strong Value", "Value", "Small Edge"]
            )
        ].copy()

        value_table = value_table.sort_values(
            "Expected ROI %",
            ascending=False
        )

        if value_table.empty:
            st.warning("No value bets found from the current odds.")
        else:
            value_table = value_table[
                [
                    "Final Tip",
                    "Match",
                    "Predicted Margin",
                    "Bookmaker Odds",
                    "Best Bookmaker",
                    "Elo Confidence",
                    "Break Even %",
                    "Model Edge %",
                    "Expected ROI %",
                    "Value Rating",
                    "Multi Eligible",
                    "Risk"
                ]
            ]

            st.dataframe(
                value_table,
                width="stretch",
                hide_index=True
            )

        # --------------------------
        # MULTI BUILDER
        # --------------------------

        st.subheader("🧩 Multi Builder")

        multi_candidates = df[
            df["Multi Eligible"] == "YES"
        ].copy()

        multi_candidates = multi_candidates.sort_values(
            "Expected ROI %",
            ascending=False
        )

        if multi_candidates.empty:
            st.warning("No multi-eligible selections found.")

        else:
            st.caption(
                "Select which value legs to include. "
                "Combined probability assumes each leg is independent, so treat it as an estimate only."
            )

            selected_rows = []

            for index, row in multi_candidates.iterrows():

                include_leg = st.checkbox(
                    f'{row["Final Tip"]} — {row["Predicted Margin"]} '
                    f'@ ${row["Bookmaker Odds"]:.2f} '
                    f'({row["Best Bookmaker"]}) | '
                    f'ROI {row["Expected ROI %"]}% | '
                    f'{row["Risk"]}',
                    value=True,
                    key=f"multi_leg_{index}"
                )

                if include_leg:
                    selected_rows.append(row)

            if not selected_rows:
                st.warning("No legs selected for the multi.")

            else:
                selected_multi = pd.DataFrame(selected_rows)

                st.dataframe(
                    selected_multi[
                        [
                            "Final Tip",
                            "Match",
                            "Predicted Margin",
                            "Bookmaker Odds",
                            "Best Bookmaker",
                            "Elo Confidence",
                            "Model Edge %",
                            "Expected ROI %",
                            "Value Rating",
                            "Risk"
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )

                combined_odds = 1
                combined_probability = 1

                for _, row in selected_multi.iterrows():
                    combined_odds *= row["Bookmaker Odds"]
                    combined_probability *= row["Elo Confidence"] / 100

                multi_break_even = 1 / combined_odds
                multi_expected_roi = (
                    combined_probability * combined_odds
                ) - 1

                leg_count = len(selected_multi)

                if leg_count <= 2:
                    multi_risk = "LOW / MEDIUM"
                elif leg_count <= 4:
                    multi_risk = "HIGH"
                else:
                    multi_risk = "VERY HIGH"

                multi_col1, multi_col2, multi_col3, multi_col4 = st.columns(4)

                with multi_col1:
                    st.metric(
                        "Selected Legs",
                        leg_count
                    )

                with multi_col2:
                    st.metric(
                        "Combined Odds",
                        f"${combined_odds:.2f}"
                    )

                with multi_col3:
                    st.metric(
                        "Estimated Hit Chance",
                        f"{combined_probability * 100:.1f}%"
                    )

                with multi_col4:
                    st.metric(
                        "Expected Multi ROI",
                        f"{multi_expected_roi * 100:.1f}%"
                    )

                st.warning(
                    f"Multi Risk: {multi_risk}. "
                    "Multi probability assumes legs are independent. "
                    "Real multis can be riskier due to correlation, team news, and variance."
                )

        # --------------------------
        # CLOSEST / BIGGEST MARGIN
        # --------------------------

        closest_game = df.loc[
            df["Raw Margin"].abs().idxmin()
        ]

        biggest_margin = df.loc[
            df["Raw Margin"].abs().idxmax()
        ]

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.info(
                f'🎯 CLOSEST GAME: '
                f'{closest_game["Predicted Margin"]} '
                f'({closest_game["Match"]})'
            )

        with summary_col2:
            st.info(
                f'💥 BIGGEST PROJECTED MARGIN: '
                f'{biggest_margin["Predicted Margin"]} '
                f'({biggest_margin["Match"]})'
            )

        # --------------------------
        # PREDICTION CARDS
        # --------------------------

        for _, row in df.iterrows():

            risk = row["Risk"]

            if risk == "LOW":
                risk_display = "🟢 LOW"
            elif risk == "MEDIUM":
                risk_display = "🟡 MEDIUM"
            else:
                risk_display = "🔴 HIGH"

            with st.container(border=True):

                st.subheader(row["Match"])

                top_col1, top_col2, top_col3, top_col4 = st.columns(4)

                with top_col1:
                    st.caption("FINAL TIP")
                    st.write(f'**{row["Final Tip"]}**')

                with top_col2:
                    st.caption("PREDICTED MARGIN")
                    st.write(f'**{row["Predicted Margin"]}**')

                with top_col3:
                    st.caption("CONFIDENCE")
                    st.write(f'**{row["Elo Confidence"]}%**')

                with top_col4:
                    st.caption("RISK")
                    st.write(f'**{risk_display}**')

                bottom_col1, bottom_col2, bottom_col3, bottom_col4, bottom_col5 = st.columns(5)

                with bottom_col1:
                    st.caption("MARGIN TYPE")
                    st.write(f'**{row["Margin Category"]}**')

                with bottom_col2:
                    st.caption("EDGE SCORE")
                    st.write(f'**{row["Edge Score"]}/10**')

                with bottom_col3:
                    st.caption("SQUIGGLE")
                    st.write(f'**{row["Squiggle Tip"]}**')
                    st.caption(f'Votes: {row["Squiggle Votes"]}')

                with bottom_col4:
                    st.caption("BEST ODDS")
                    st.write(f'**${row["Bookmaker Odds"]:.2f}**')
                    st.caption(
                        f'{row["Best Bookmaker"]} | '
                        f'Break-even: {row["Break Even %"]}%'
                    )

                with bottom_col5:
                    st.caption("VALUE")
                    st.write(f'**{row["Value Rating"]}**')
                    st.caption(
                        f'Edge: {row["Model Edge %"]}% | '
                        f'ROI: {row["Expected ROI %"]}%'
                    )

        # --------------------------
        # PICK GROUPS
        # --------------------------

        safe = df[df["Risk"] == "LOW"]
        solid = df[df["Risk"] == "MEDIUM"]
        danger = df[df["Risk"] == "HIGH"]

        group_col1, group_col2, group_col3 = st.columns(3)

        with group_col1:
            st.subheader("🔥 Best Picks")

            if safe.empty:
                st.write("No low-risk picks found.")
            else:
                for _, row in safe.iterrows():
                    st.success(
                        f'{row["Predicted Margin"]} | '
                        f'{row["Elo Confidence"]}% | '
                        f'Edge {row["Edge Score"]}/10'
                    )

        with group_col2:
            st.subheader("🟡 Solid Picks")

            if solid.empty:
                st.write("No medium-risk picks found.")
            else:
                for _, row in solid.iterrows():
                    st.info(
                        f'{row["Predicted Margin"]} | '
                        f'{row["Elo Confidence"]}% | '
                        f'Edge {row["Edge Score"]}/10'
                    )

        with group_col3:
            st.subheader("⚠️ Danger Games")

            if danger.empty:
                st.write("No high-risk games.")
            else:
                for _, row in danger.iterrows():
                    st.warning(
                        f'{row["Match"]} → '
                        f'{row["Predicted Margin"]} | '
                        f'{row["Elo Confidence"]}%'
                    )

        # --------------------------
        # CSV EXPORT
        # --------------------------

        os.makedirs("outputs", exist_ok=True)

        output_path = f"outputs/round_{round_number}_app_tips.csv"

        df.to_csv(output_path, index=False)

        st.success(f"Saved CSV to {output_path}")

        # --------------------------
        # BET TRACKER EXPORT
        # --------------------------

        bet_tracker = df[
            df["Value Rating"].isin(
                ["Strong Value", "Value", "Small Edge"]
            )
        ].copy()

        bet_tracker["Round"] = round_number
        bet_tracker["Bet Type"] = "H2H"
        bet_tracker["Recommended Units"] = 1
        bet_tracker["Recommended Stake"] = unit_size
        bet_tracker["Stake"] = ""
        bet_tracker["Result"] = ""
        bet_tracker["Profit/Loss"] = ""
        bet_tracker["Notes"] = ""

        bet_tracker = bet_tracker[
            [
                "Round",
                "Match",
                "Final Tip",
                "Bet Type",
                "Predicted Margin",
                "Bookmaker Odds",
                "Best Bookmaker",
                "Elo Confidence",
                "Break Even %",
                "Model Edge %",
                "Expected ROI %",
                "Value Rating",
                "Multi Eligible",
                "Risk",
                "Recommended Units",
                "Recommended Stake",
                "Stake",
                "Result",
                "Profit/Loss",
                "Notes"
            ]
        ]

        tracker_path = f"outputs/round_{round_number}_bet_tracker.csv"

        bet_tracker.to_csv(
            tracker_path,
            index=False
        )

        st.success(
            f"Saved Bet Tracker CSV to {tracker_path}"
        )

        st.download_button(
            label="Download Bet Tracker CSV",
            data=bet_tracker.to_csv(index=False),
            file_name=f"round_{round_number}_bet_tracker.csv",
            mime="text/csv"
        )