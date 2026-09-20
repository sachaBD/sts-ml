---
name: pi-intercom
description: >-
  Communicate with peer agents (such as terra-impl and sol-coord) running in the local Pi agent harness via pi-intercom. Use this skill when coordinating across agent harnesses, sending messages or replies to peer agents, checking active peer sessions, or arming the background listener for auto wake-up.
---

# Pi Intercom Communication Skill

This skill enables Antigravity to communicate directly with local Pi agent sessions running in the same or related repositories over the `pi-intercom` broker.

## CLI & Helper Path

The bridge CLI is installed at `~/.local/bin/agy-intercom` (source: `scripts/intercom_bridge.js`).

## Quick Commands

### 1. Check Active Peer Agents
```bash
agy-intercom list
```
Shows connected sessions, models, and current activity status (e.g. `terra-impl`, `sol-coord`, `antigravity`).

### 2. Send a Message to an Agent
Fire-and-forget message:
```bash
agy-intercom send <target> "<message>"
```
Example:
```bash
agy-intercom send terra-impl "Completed scenario tests in python/sim. 4 passing."
```

### 3. Reply to an Inbound Ask
When a peer agent asks a question and is waiting for a response, include the `replyTo` message ID:
```bash
agy-intercom send <target> "<answer>" --reply-to <messageId>
```
Example:
```bash
agy-intercom send terra-impl "Use PPO for the multi-enemy scenarios." --reply-to c4ccf6db-70bb-4167-ba48-a4aa8a3d9bc7
```

### 4. Blocking Ask from Antigravity to Pi
If you need an answer before proceeding:
```bash
agy-intercom send <target> "<question>" --ask
```

---

## Automatic Wake-Up (Background Listener Pattern)

Whenever Antigravity completes a task or turn and wants to wait for the next message from Pi:

1. **Launch `agy-intercom wait-message` as a background task**:
   Call `run_command` with:
   - `CommandLine`: `agy-intercom wait-message`
   - `WaitMsBeforeAsync`: `500`

2. **End turn**:
   Inform the user you are listening and stop calling tools. Antigravity becomes idle.

3. **Reactive Wake-Up**:
   The moment a message is sent to `antigravity` from any Pi session, the background task completes with exit code 0. Antigravity is immediately resumed with high priority, with the sender, metadata, and full message text injected into the turn.

4. **Process and Re-arm**:
   Process the message, send any required reply with `--reply-to`, and call `agy-intercom wait-message` again if further messages are expected.
