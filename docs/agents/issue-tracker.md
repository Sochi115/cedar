# Issue tracker: Local Markdown

Issues live as Markdown files under `.scratch/tickets/`.

## Conventions

- One issue per file: `.scratch/tickets/<feature>.md`
- Status is recorded near the top with a `Status:` line.
- Triage labels use strings defined in `triage-labels.md`.
- Comments append under a `## Comments` heading.

## Wayfinding operations

- Map: `.scratch/tickets/<effort>-map.md`
- Child ticket: `.scratch/tickets/<effort>-<NN>-<slug>.md`
- Ticket type: `research`, `prototype`, `grilling`, or `task`
- Open ticket status: `open` or `claimed`
- Resolved ticket status: `resolved`
- Blocking uses a `Blocked by: <ticket filenames>` line.
- A ticket is unblocked when every blocking ticket has `Status: resolved`.
- Frontier: open, unblocked, unclaimed tickets, ordered by filename.
- Claim: set `Status: claimed` before work.
- Resolve: append an `## Answer` section, set `Status: resolved`, then update the map's `Decisions so far` section.
