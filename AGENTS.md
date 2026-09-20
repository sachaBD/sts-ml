# Project Agent Collaboration & Memory

## Cross-Agent Intercom Bridge

This repository participates in multi-agent collaboration between **Google Antigravity** (`antigravity` / Gemini Flash) and **Pi agent sessions** (e.g. `sol-coord`, `terra-impl`) via `pi-intercom`.

### For Antigravity Agents:
- **Skill**: Use `/skill:pi-intercom` or read `.agents/skills/pi-intercom/SKILL.md`.
- **CLI tool**: `agy-intercom` (installed at `~/.local/bin/agy-intercom`).
- **Auto Wake-Up**: Run `agy-intercom wait-message` as a background task (`WaitMsBeforeAsync: 500`) whenever waiting for instructions, tasks, or answers from Pi peers.

### For Pi Agents:
- Target Antigravity by name: `to: "antigravity"`.
- Use `intercom({ action: "send", to: "antigravity", message: "..." })` for notifications/tasks.
- Use `intercom({ action: "ask", to: "antigravity", message: "..." })` for blocking questions.
