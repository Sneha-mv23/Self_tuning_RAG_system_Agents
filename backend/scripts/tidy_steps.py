import sys

from app.db.experiments import connect

conn = connect()
if len(sys.argv) == 4 and sys.argv[1] == "--repoint":
    old, new = int(sys.argv[2]), int(sys.argv[3])
    conn.execute("UPDATE optimizer_steps SET candidate_run_id = ? WHERE candidate_run_id = ?", (new, old))
    print(f"repointed steps from run {old} to run {new}")

n = conn.execute(
    "DELETE FROM optimizer_steps WHERE decision = 'skipped_screen' AND session NOT IN "
    "(SELECT session FROM optimizer_steps WHERE decision IN ('accepted', 'rejected', 'aborted'))"
).rowcount
existing = {r["id"] for r in conn.execute("SELECT id FROM runs")}
dangling = [(r["id"], r["candidate_run_id"], r["decision"]) for r in conn.execute(
    "SELECT id, candidate_run_id, decision FROM optimizer_steps WHERE candidate_run_id IS NOT NULL")
    if r["candidate_run_id"] not in existing]
conn.commit()
print(f"deleted {n} skip rows left by dry runs (sessions where no experiment ran)")
print(f"steps pointing at a run that no longer exists (step id, run id, decision): {dangling}")