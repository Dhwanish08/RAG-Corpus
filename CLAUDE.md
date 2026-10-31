Read PROGRESS.md first. It is the source of truth for what's done and what's next.
Update PROGRESS.md at the end of every prompt that implements something (run log, change log, next action). Do not record the user's questions or discussion-only prompts; record only the implementations they ask for. Use it as the source of truth for where things stand. Work one step at a time and tell the user what's done and what's next.
Run scripts from `r2/` (paths are relative). The defense must never read the `poison` flag; it is for metrics only.
