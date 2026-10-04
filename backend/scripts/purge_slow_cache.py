import sqlite3

conn = sqlite3.connect("data/cache/generations.sqlite")
n = conn.execute("DELETE FROM generations WHERE latency_s > 600").rowcount
conn.commit()
print(f"deleted {n} stalled entries")