import os
import json
from collections import defaultdict

# CRICSHEET

# Path where all Cricsheet match JSON files are stored
MATCHES_DIR = "cricsheet/"   # change if needed

# Player stats storage
players = defaultdict(lambda: {
    "matches": 0,
    "innings_batted": 0,
    "not_outs": 0,
    "runs": 0,
    "balls_faced": 0,
    "fours": 0,
    "sixes": 0,
    "ducks": 0,
    "high_score": 0,

    "wickets": 0,
    "balls_bowled": 0,
    "runs_conceded": 0,
})


def process_match(match_file):
    with open(match_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep track of players who batted in this match
    match_batted = set()
    match_bowled = set()
    match_players = set()

    # Cricsheet structure:
    # data["innings"] -> list of innings
    for inning in data.get("innings", []):
        for innings_name, innings_data in inning.items():
            for delivery in innings_data.get("deliveries", []):
                for ball, info in delivery.items():
                    batsman = info["batsman"]
                    bowler = info["bowler"]

                    match_players.update([batsman, bowler])

                    # ------- Batting Stats --------
                    runs = info["runs"]["batsman"]
                    players[batsman]["runs"] += runs
                    players[batsman]["balls_faced"] += 1

                    if runs == 4:
                        players[batsman]["fours"] += 1
                    if runs == 6:
                        players[batsman]["sixes"] += 1

                    match_batted.add(batsman)

                    # Dismissal
                    if "wicket" in info:
                        wicket = info["wicket"]
                        if wicket.get("player_out") == batsman:
                            # Duck check
                            if players[batsman]["runs"] == 0:
                                players[batsman]["ducks"] += 1

                            # this is an out
                            players[batsman]["innings_batted"] += 1
                        else:
                            # runout or other wicket on non-striker (ignore)
                            pass

                    # ------- Bowling Stats --------
                    players[bowler]["balls_bowled"] += 1
                    players[bowler]["runs_conceded"] += info["runs"]["total"]

                    if "wicket" in info:
                        wicket = info["wicket"]
                        if wicket.get("kind") not in ["run out", "retired hurt"]:
                            if wicket.get("player_out") != bowler:
                                players[bowler]["wickets"] += 1

                    match_bowled.add(bowler)

    # Increase match count for all who appeared
    for p in match_players:
        players[p]["matches"] += 1

    # Count not-outs
    for p in match_batted:
        players[p]["innings_batted"] += 1  # default assumption
    # But if player is not in dismissal list → not out
    # Already handled above


def compute_final_stats():
    final_stats = {}

    for player, stats in players.items():
        if stats["matches"] == 0:
            continue

        # Batting average
        outs = stats["innings_batted"] - stats["not_outs"]
        average = stats["runs"] / outs if outs > 0 else stats["runs"]

        # Strike rate
        strike_rate = (stats["runs"] / stats["balls_faced"] * 100) if stats["balls_faced"] > 0 else 0

        # Bowling economy
        overs = stats["balls_bowled"] / 6
        economy = stats["runs_conceded"] / overs if overs > 0 else 0

        final_stats[player] = {
            "matches": stats["matches"],

            # Batting
            "runs": stats["runs"],
            "balls_faced": stats["balls_faced"],
            "average": round(average, 2),
            "strike_rate": round(strike_rate, 2),
            "fours": stats["fours"],
            "sixes": stats["sixes"],
            "ducks": stats["ducks"],

            # Bowling
            "wickets": stats["wickets"],
            "balls_bowled": stats["balls_bowled"],
            "runs_conceded": stats["runs_conceded"],
            "economy": round(economy, 2),
        }

    return final_stats


def main():
    print("Processing Cricsheet IPL matches...")
    files = [f for f in os.listdir(MATCHES_DIR) if f.endswith(".json")]

    for i, file in enumerate(files, start=1):
        filepath = os.path.join(MATCHES_DIR, file)
        process_match(filepath)
        if i % 50 == 0:
            print(f"Processed {i}/{len(files)} matches...")

    print("Computing final stats...")
    stats = compute_final_stats()

    with open("ipl_player_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("Done! Saved to ipl_player_stats.json")


if __name__ == "__main__":
    main()
