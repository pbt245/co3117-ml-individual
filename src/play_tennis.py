"""The Play Tennis toy dataset in Decision_Tree.pdf (14 days)."""

COLUMNS = ["Outlook", "Temp", "Humidity", "Wind", "Play"]

_ROWS = """\
Sunny Hot High Weak No
Sunny Hot High Strong No
Overcast Hot High Weak Yes
Rain Mild High Weak Yes
Rain Cool Normal Weak Yes
Rain Cool Normal Strong No
Overcast Cool Normal Strong Yes
Sunny Mild High Weak No
Sunny Cool Normal Weak Yes
Rain Mild Normal Weak Yes
Sunny Mild Normal Strong Yes
Overcast Mild High Strong Yes
Overcast Hot Normal Weak Yes
Rain Mild High Strong No"""

PLAY_TENNIS = [dict(zip(COLUMNS, line.split())) for line in _ROWS.splitlines()]
