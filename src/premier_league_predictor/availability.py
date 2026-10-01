"""Public FPL availability snapshots; context only until historical data exists."""

from datetime import datetime, timezone
from pathlib import Path
import json
from urllib.request import Request, urlopen

SOURCE = "https://fantasy.premierleague.com/api/bootstrap-static/"
ALIASES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nott'm Forest",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
}


def refresh_availability(path):
    with urlopen(
        Request(SOURCE, headers={"User-Agent": "Matchday/1.0"}), timeout=30
    ) as response:
        payload = json.load(response)
    teams = {
        team["id"]: ALIASES.get(team["name"], team["name"]) for team in payload["teams"]
    }
    rows = [
        {
            "player": f"{player['first_name']} {player['second_name']}",
            "team": teams[player["team"]],
            "position": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(
                player["element_type"], "Unknown"
            ),
            "status": player["status"],
            "news": player["news"],
            "minutes": player["minutes"],
            "chance_of_playing": player.get("chance_of_playing_next_round"),
        }
        for player in payload["elements"]
    ]
    snapshot = {
        "source": SOURCE,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "players": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(path)
    print(f"Saved public availability for {len(rows)} players")
    return snapshot


def availability_for(path, home, away, kickoff, now):
    if not Path(path).exists():
        return {"status": "unavailable", "players": []}
    snapshot = json.loads(Path(path).read_text())
    observed = datetime.fromisoformat(snapshot["observed_at"])
    if (
        observed > now
        or (now - observed).total_seconds() > 48 * 3600
        or (kickoff - now).total_seconds() > 7 * 86400
    ):
        return {"status": "not_current_for_fixture", "players": []}
    players = [
        row
        for row in snapshot["players"]
        if row["team"] in (home, away) and row["status"] in ("i", "s", "d")
    ]
    return {
        "status": "context_only",
        "observed_at": snapshot["observed_at"],
        "players": players,
        "source": SOURCE,
    }
