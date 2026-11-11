"""Research sandbox for FRIDAY's MODEL track (docs/architecture/19).

Nothing here is on the agent's import path. The core stays stdlib + sqlite3;
this package is where architecture innovations get an EXECUTABLE contract —
pure-Python reference implementations that produce golden vectors — before any
GPU kernel is allowed to exist. A Triton kernel you can't test against a
hand-checked reference is a vibe, not an operator.
"""
