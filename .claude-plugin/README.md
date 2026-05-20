# .claude-plugin/

This directory is metadata-only — it declares the `lingxi-loop` plugin for
future packaging / marketplace publication. **The hooks themselves are
activated through `.claude/settings.json`, not this directory.**

## Why two locations

Claude Code discovers hooks two ways:

| Location | Auto-activate on clone? | Use for |
|---|---|---|
| `.claude/settings.json` (this repo) | ✅ Yes | Project-local hooks, committed to git |
| `.claude-plugin/plugin.json` + `hooks/hooks.json` | ❌ No, needs `/plugin install` | Distributable plugins |

We want clone-and-go behavior, so hooks live in `.claude/settings.json`. This
directory is kept so we can later run `/plugin install <repo>` to publish the
same hooks as a portable plugin without restructuring.

## Files

- `plugin.json` — plugin metadata (name, version, description)
- (future) `hooks.json` — hook registration in plugin form, mirroring `.claude/settings.json`

## Hook scripts

The actual hook scripts live in `.claude/hooks/`:

```
.claude/hooks/
├── loop-bash-safety.sh   # PreToolUse(Bash) — path safety (B1/B2 rules)
├── loop-stop.sh          # Stop — state/artifact consistency (R2-R5 rules; Phase C)
└── lib/
    ├── common.sh         # shared helpers (logging, JSON, state.json lookup)
    ├── paths.sh          # path safety logic shared by Bash + Stop hooks
    └── state.sh          # state.json rule checks for Stop hook
```

See `evolution/world_model/state_schema.md` for the runtime state machine
that the hooks validate against.
