---
name: insight-design
description: Build compact, evidence-aware behavioral insights from event tables.
tools:
  - execute_python_code
  - read_table
triggers:
  - client behavior
  - repeated events
  - data insight
---

# Procedure

1. Identify the current subject, time window, and event type.
2. Look for repeated event attributes: recipient, merchant, amount band, channel, time of day, and stop status.
3. Separate observed facts from interpretation.
4. State limitations clearly when data is a small sample or missing historical depth.
5. Produce a compact insight with evidence pointers or artifact references when available.

# Output Style

Use this structure:

```text
Insight:
Evidence:
Limitations:
Next checks:
```
