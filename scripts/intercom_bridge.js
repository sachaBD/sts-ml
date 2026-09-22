#!/usr/bin/env node

/**
 * Antigravity <-> Pi Intercom Bridge
 * 
 * Bridges Google Antigravity sessions with Pi agent sessions over
 * the local pi-intercom broker Unix domain socket (~/.pi/agent/intercom/broker.sock).
 */

const net = require("net");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const { spawn } = require("child_process");

const BROKER_SOCK = process.env.PI_INTERCOM_SOCKET || path.join(os.homedir(), ".pi/agent/intercom/broker.sock");
const RUNTIME_DIR = process.env.INTERCOM_RUNTIME_DIR || path.join(os.homedir(), ".local/share/antigravity-intercom");
const AGENT_NAME = process.env.INTERCOM_AGENT_NAME || process.env.ANTIGRAVITY_AGENT_NAME || "antigravity";
const AGENT_MODEL = process.env.INTERCOM_AGENT_MODEL || "gemini-3.8-flash";
const BRIDGE_SOCK = path.join(RUNTIME_DIR, "bridge.sock");
const BRIDGE_PID = path.join(RUNTIME_DIR, "bridge.pid");
const INBOX_FILE = path.join(RUNTIME_DIR, "inbox.jsonl");

// Framing helpers for broker (4-byte big-endian uint32 length + JSON)
function writeBrokerMessage(socket, obj) {
  const json = Buffer.from(JSON.stringify(obj), "utf8");
  const frame = Buffer.allocUnsafe(4 + json.length);
  frame.writeUInt32BE(json.length, 0);
  json.copy(frame, 4);
  socket.write(frame);
}

function createBrokerReader(onMessage, onError) {
  let buf = Buffer.alloc(0);
  return (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= 4) {
      const len = buf.readUInt32BE(0);
      if (buf.length < 4 + len) break;
      const jsonBuf = buf.subarray(4, 4 + len);
      buf = buf.subarray(4 + len);
      try {
        const msg = JSON.parse(jsonBuf.toString("utf8"));
        onMessage(msg);
      } catch (err) {
        if (onError) onError(err);
      }
    }
  };
}

class IntercomDaemon {
  constructor() {
    this.brokerSocket = null;
    this.sessionId = null;
    this.server = null;
    this.senderSeq = 1;
    this.inbox = [];
    this.unread = [];
    this.waiters = []; // { resolve, reject, timer }
    this.pendingBrokerRequests = new Map(); // requestId -> { resolve, reject, timer }
    this.connected = false;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.agentName = AGENT_NAME;
    this.agentModel = AGENT_MODEL;
    this.agentCwd = process.cwd();
  }

  start() {
    if (!fs.existsSync(RUNTIME_DIR)) {
      fs.mkdirSync(RUNTIME_DIR, { recursive: true });
    }

    // Write PID file
    fs.writeFileSync(BRIDGE_PID, String(process.pid));

    // Load any existing unread messages from inbox file
    this.loadInbox();

    // Start local bridge server
    this.startBridgeServer();

    // Connect to Pi broker
    this.connectToBroker();

    // Handle shutdown signals
    const cleanup = () => {
      this.shutdown();
      process.exit(0);
    };
    process.on("SIGINT", cleanup);
    process.on("SIGTERM", cleanup);
    process.on("exit", () => this.shutdown());
  }

  loadInbox() {
    if (fs.existsSync(INBOX_FILE)) {
      try {
        const lines = fs.readFileSync(INBOX_FILE, "utf8").split("\n").filter(Boolean);
        for (const line of lines) {
          const item = JSON.parse(line);
          this.inbox.push(item);
          if (!item.read) {
            this.unread.push(item);
          }
        }
      } catch (e) {
        console.error("Error reading inbox file:", e.message);
      }
    }
  }

  saveInboxItem(item) {
    this.inbox.push(item);
    this.unread.push(item);
    try {
      fs.appendFileSync(INBOX_FILE, JSON.stringify(item) + "\n");
    } catch (e) {
      console.error("Error appending to inbox:", e.message);
    }
  }

  startBridgeServer() {
    if (fs.existsSync(BRIDGE_SOCK)) {
      try {
        fs.unlinkSync(BRIDGE_SOCK);
      } catch {}
    }

    this.server = net.createServer((client) => {
      let lineBuf = "";
      client.on("data", (chunk) => {
        lineBuf += chunk.toString("utf8");
        let idx;
        while ((idx = lineBuf.indexOf("\n")) !== -1) {
          const line = lineBuf.substring(0, idx).trim();
          lineBuf = lineBuf.substring(idx + 1);
          if (line) {
            this.handleClientCommand(client, line);
          }
        }
      });
    });

    this.server.listen(BRIDGE_SOCK, () => {
      fs.chmodSync(BRIDGE_SOCK, 0o600);
      console.log(`[Bridge] Local bridge socket ready at ${BRIDGE_SOCK}`);
    });
  }

  async handleClientCommand(client, line) {
    try {
      const req = JSON.parse(line);
      const action = req.action;

      if (action === "status") {
        client.write(JSON.stringify({
          ok: true,
          connected: this.connected,
          sessionId: this.sessionId,
          agentName: this.agentName,
          brokerSocket: BROKER_SOCK,
          unreadCount: this.unread.length,
          totalInbox: this.inbox.length
        }) + "\n");
        client.end();
      } else if (action === "list") {
        if (!this.connected) {
          client.write(JSON.stringify({ ok: false, error: "Not connected to broker" }) + "\n");
          client.end();
          return;
        }
        const sessions = await this.listBrokerSessions();
        client.write(JSON.stringify({ ok: true, sessions }) + "\n");
        client.end();
      } else if (action === "send") {
        if (!this.connected) {
          client.write(JSON.stringify({ ok: false, error: "Not connected to broker" }) + "\n");
          client.end();
          return;
        }
        const result = await this.sendToBroker(req.to, req.text, {
          replyTo: req.replyTo,
          expectsReply: req.expectsReply
        });
        client.write(JSON.stringify({ ok: true, result }) + "\n");
        client.end();
      } else if (action === "wait_message") {
        // If there's an unread message already waiting, return it immediately
        if (this.unread.length > 0) {
          const item = this.unread.shift();
          item.read = true;
          this.persistInboxUpdate();
          client.write(JSON.stringify({ ok: true, message: item }) + "\n");
          client.end();
          return;
        }

        // Otherwise enqueue waiter
        const timeoutMs = req.timeout || 0; // 0 = wait indefinitely
        let timer = null;

        const waiter = {
          resolve: (item) => {
            if (timer) clearTimeout(timer);
            try {
              client.write(JSON.stringify({ ok: true, message: item }) + "\n");
              client.end();
            } catch {}
          },
          reject: (err) => {
            if (timer) clearTimeout(timer);
            try {
              client.write(JSON.stringify({ ok: false, error: err.message }) + "\n");
              client.end();
            } catch {}
          }
        };

        if (timeoutMs > 0) {
          timer = setTimeout(() => {
            const idx = this.waiters.indexOf(waiter);
            if (idx !== -1) this.waiters.splice(idx, 1);
            waiter.reject(new Error("Timeout waiting for message"));
          }, timeoutMs);
        }

        this.waiters.push(waiter);

        client.on("close", () => {
          if (timer) clearTimeout(timer);
          const idx = this.waiters.indexOf(waiter);
          if (idx !== -1) this.waiters.splice(idx, 1);
        });
      } else if (action === "inbox") {
        client.write(JSON.stringify({ ok: true, messages: this.inbox, unread: this.unread }) + "\n");
        client.end();
      } else {
        client.write(JSON.stringify({ ok: false, error: `Unknown action: ${action}` }) + "\n");
        client.end();
      }
    } catch (err) {
      try {
        client.write(JSON.stringify({ ok: false, error: err.message }) + "\n");
        client.end();
      } catch {}
    }
  }

  persistInboxUpdate() {
    try {
      const content = this.inbox.map(item => JSON.stringify(item)).join("\n") + "\n";
      fs.writeFileSync(INBOX_FILE, content);
    } catch (e) {
      console.error("Error writing inbox file:", e.message);
    }
  }

  connectToBroker() {
    if (this.brokerSocket) {
      try { this.brokerSocket.destroy(); } catch {}
      this.brokerSocket = null;
    }

    console.log(`[Bridge] Connecting to Pi Intercom Broker at ${BROKER_SOCK}...`);
    const sock = net.connect(BROKER_SOCK);
    this.brokerSocket = sock;

    const reader = createBrokerReader(
      (msg) => this.handleBrokerMessage(msg),
      (err) => console.error("[Bridge] Broker parse error:", err.message)
    );

    sock.on("connect", () => {
      console.log("[Bridge] Connected to Pi Intercom Broker, sending registration...");
      writeBrokerMessage(sock, {
        type: "register",
        session: {
          cwd: this.agentCwd,
          model: this.agentModel,
          pid: process.pid,
          startedAt: Date.now(),
          lastActivity: Date.now(),
          name: this.agentName
        }
      });
    });

    sock.on("data", reader);

    sock.on("error", (err) => {
      console.error("[Bridge] Broker socket error:", err.message);
    });

    sock.on("close", () => {
      console.log("[Bridge] Broker socket closed. Reconnecting in 3s...");
      this.connected = false;
      this.sessionId = null;
      if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
      this.scheduleReconnect();
    });
  }

  scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this.connectToBroker();
    }, 3000);
  }

  startHeartbeat() {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = setInterval(async () => {
      if (this.connected && this.brokerSocket) {
        try {
          await this.listBrokerSessions();
        } catch {}
      }
    }, 25000);
  }

  handleBrokerMessage(msg) {
    if (!msg || typeof msg !== "object") return;

    if (msg.type === "registered") {
      this.sessionId = msg.sessionId;
      this.connected = true;
      console.log(`[Bridge] Successfully registered as '${this.agentName}' (ID: ${this.sessionId})`);
      this.startHeartbeat();
    } else if (msg.type === "sessions") {
      const pending = this.pendingBrokerRequests.get(msg.requestId);
      if (pending) {
        this.pendingBrokerRequests.delete(msg.requestId);
        pending.resolve(msg.sessions);
      }
    } else if (msg.type === "delivered") {
      const pending = this.pendingBrokerRequests.get(msg.messageId);
      if (pending) {
        this.pendingBrokerRequests.delete(msg.messageId);
        pending.resolve(msg);
      }
    } else if (msg.type === "delivery_failed") {
      const pending = this.pendingBrokerRequests.get(msg.messageId);
      if (pending) {
        this.pendingBrokerRequests.delete(msg.messageId);
        pending.reject(new Error(msg.reason || "Delivery failed"));
      }
    } else if (msg.type === "message") {
      this.handleIncomingMessage(msg.from, msg.message);
    }
  }

  handleIncomingMessage(from, message) {
    console.log(`[Bridge] Incoming message from ${from.name || from.id}: "${message.content.text}"`);

    // Acknowledge receipt to broker
    try {
      writeBrokerMessage(this.brokerSocket, {
        type: "message_receipt",
        receipt: {
          messageId: message.id,
          status: "receiver_received",
          timestamp: Date.now()
        }
      });
      writeBrokerMessage(this.brokerSocket, {
        type: "message_receipt",
        receipt: {
          messageId: message.id,
          status: "acknowledged",
          timestamp: Date.now(),
          detail: `accepted by ${this.agentName} bridge`
        }
      });
    } catch (e) {
      console.error("[Bridge] Failed to send receipt:", e.message);
    }

    const inboxItem = {
      id: message.id,
      receivedAt: Date.now(),
      from: {
        id: from.id,
        name: from.name || from.id,
        cwd: from.cwd,
        model: from.model
      },
      content: message.content.text,
      attachments: message.content.attachments || [],
      replyTo: message.replyTo || null,
      expectsReply: Boolean(message.expectsReply),
      read: false
    };

    // If there is an active waiter (wait-message command running), fulfill it immediately!
    if (this.waiters.length > 0) {
      const waiter = this.waiters.shift();
      inboxItem.read = true;
      this.saveInboxItem(inboxItem);
      waiter.resolve(inboxItem);
    } else {
      // Otherwise store in unread queue
      this.saveInboxItem(inboxItem);
    }
  }

  listBrokerSessions() {
    return new Promise((resolve, reject) => {
      if (!this.connected || !this.brokerSocket) {
        return reject(new Error("Not connected to broker"));
      }
      const requestId = crypto.randomUUID();
      const timer = setTimeout(() => {
        this.pendingBrokerRequests.delete(requestId);
        reject(new Error("List sessions timeout"));
      }, 5000);

      this.pendingBrokerRequests.set(requestId, {
        resolve: (sessions) => {
          clearTimeout(timer);
          resolve(sessions);
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        }
      });

      writeBrokerMessage(this.brokerSocket, {
        type: "list",
        requestId
      });
    });
  }

  sendToBroker(to, text, options = {}) {
    return new Promise((resolve, reject) => {
      if (!this.connected || !this.brokerSocket) {
        return reject(new Error("Not connected to broker"));
      }

      const messageId = crypto.randomUUID();
      const timer = setTimeout(() => {
        this.pendingBrokerRequests.delete(messageId);
        reject(new Error("Send timeout"));
      }, 10000);

      this.pendingBrokerRequests.set(messageId, {
        resolve: (res) => {
          clearTimeout(timer);
          resolve(res);
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        }
      });

      const message = {
        id: messageId,
        timestamp: Date.now(),
        senderSequence: this.senderSeq++,
        replyTo: options.replyTo || undefined,
        expectsReply: Boolean(options.expectsReply),
        content: {
          text: text,
          attachments: []
        }
      };

      writeBrokerMessage(this.brokerSocket, {
        type: "send",
        to,
        message
      });
    });
  }

  shutdown() {
    console.log("[Bridge] Shutting down...");
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.brokerSocket) {
      try {
        writeBrokerMessage(this.brokerSocket, { type: "unregister" });
        this.brokerSocket.end();
      } catch {}
      this.brokerSocket = null;
    }
    if (this.server) {
      try { this.server.close(); } catch {}
      this.server = null;
    }
    if (fs.existsSync(BRIDGE_SOCK)) {
      try { fs.unlinkSync(BRIDGE_SOCK); } catch {}
    }
    if (fs.existsSync(BRIDGE_PID)) {
      try { fs.unlinkSync(BRIDGE_PID); } catch {}
    }
  }
}

// ---------------------------------------------------------------------------
// Client CLI helper methods
// ---------------------------------------------------------------------------

function sendBridgeRequest(req) {
  return new Promise((resolve, reject) => {
    const sock = net.connect(BRIDGE_SOCK);
    let data = "";

    sock.on("connect", () => {
      sock.write(JSON.stringify(req) + "\n");
    });

    sock.on("data", (chunk) => {
      data += chunk.toString("utf8");
    });

    sock.on("end", () => {
      try {
        const res = JSON.parse(data.trim());
        resolve(res);
      } catch (err) {
        reject(new Error(`Failed to parse bridge response: ${data}`));
      }
    });

    sock.on("error", (err) => {
      reject(err);
    });
  });
}

async function isBridgeRunning() {
  if (!fs.existsSync(BRIDGE_SOCK)) return false;
  try {
    const res = await sendBridgeRequest({ action: "status" });
    return res && res.ok;
  } catch {
    return false;
  }
}

async function ensureBridgeRunning() {
  if (await isBridgeRunning()) return true;

  if (!fs.existsSync(RUNTIME_DIR)) {
    fs.mkdirSync(RUNTIME_DIR, { recursive: true });
  }

  console.log("[Bridge CLI] Bridge daemon not running, starting in background...");
  const scriptPath = __filename;
  const outLog = fs.openSync(path.join(RUNTIME_DIR, "bridge.log"), "a");
  const errLog = fs.openSync(path.join(RUNTIME_DIR, "bridge.log"), "a");

  const child = spawn(process.execPath, [scriptPath, "daemon"], {
    detached: true,
    stdio: ["ignore", outLog, errLog]
  });
  child.unref();

  // Wait up to 5s for bridge to come up
  for (let i = 0; i < 25; i++) {
    await new Promise(r => setTimeout(r, 200));
    if (await isBridgeRunning()) {
      console.log("[Bridge CLI] Bridge daemon is now active.");
      return true;
    }
  }

  throw new Error(`Timed out waiting for bridge daemon to start. Check ${path.join(RUNTIME_DIR, "bridge.log")}`);
}

// ---------------------------------------------------------------------------
// Main CLI Entrypoint
// ---------------------------------------------------------------------------

async function main() {
  const args = process.argv.slice(2);
  const cmd = args[0] || "help";

  if (cmd === "daemon") {
    const daemon = new IntercomDaemon();
    daemon.start();
    return;
  }

  if (cmd === "start") {
    await ensureBridgeRunning();
    const status = await sendBridgeRequest({ action: "status" });
    console.log("Bridge daemon running:", status);
    return;
  }

  if (cmd === "stop") {
    if (fs.existsSync(BRIDGE_PID)) {
      const pid = parseInt(fs.readFileSync(BRIDGE_PID, "utf8").trim(), 10);
      try {
        process.kill(pid, "SIGTERM");
        console.log(`Sent SIGTERM to bridge daemon (PID ${pid})`);
      } catch (e) {
        console.log(`Could not kill PID ${pid}: ${e.message}`);
      }
      try { fs.unlinkSync(BRIDGE_PID); } catch {}
      try { fs.unlinkSync(BRIDGE_SOCK); } catch {}
    } else {
      console.log("No bridge PID file found.");
    }
    return;
  }

  if (cmd === "status") {
    await ensureBridgeRunning();
    const res = await sendBridgeRequest({ action: "status" });
    console.log(JSON.stringify(res, null, 2));
    return;
  }

  if (cmd === "list") {
    await ensureBridgeRunning();
    const res = await sendBridgeRequest({ action: "list" });
    if (!res.ok) {
      console.error("Failed to list sessions:", res.error);
      process.exit(1);
    }
    console.log(`\nActive Pi Intercom Sessions (${res.sessions.length}):`);
    console.log("--------------------------------------------------------------------------------");
    for (const s of res.sessions) {
      const isSelf = s.name === AGENT_NAME;
      const tag = isSelf ? " (you)" : "";
      console.log(`• ${s.name || s.id}${tag}`);
      console.log(`  ID:     ${s.id}`);
      console.log(`  Model:  ${s.model}`);
      console.log(`  CWD:    ${s.cwd}`);
      console.log(`  Status: ${s.status || "idle"}`);
      console.log("");
    }
    return;
  }

  if (cmd === "send") {
    const to = args[1];
    const text = args[2];
    if (!to || !text) {
      console.error("Usage: agy-intercom send <target> <message> [--reply-to <id>] [--ask]");
      process.exit(1);
    }

    let replyTo = undefined;
    let expectsReply = false;
    for (let i = 3; i < args.length; i++) {
      if (args[i] === "--reply-to" && args[i + 1]) {
        replyTo = args[i + 1];
        i++;
      } else if (args[i] === "--ask") {
        expectsReply = true;
      }
    }

    await ensureBridgeRunning();
    const res = await sendBridgeRequest({
      action: "send",
      to,
      text,
      replyTo,
      expectsReply
    });

    if (!res.ok) {
      console.error("Send failed:", res.error);
      process.exit(1);
    }
    console.log(`✓ Message sent to "${to}" (ID: ${res.result.messageId || "ok"})`);
    return;
  }

  if (cmd === "wait-message") {
    let timeout = 0;
    if (args[1] === "--timeout" && args[2]) {
      timeout = parseInt(args[2], 10);
    }

    await ensureBridgeRunning();
    const res = await sendBridgeRequest({ action: "wait_message", timeout });

    if (!res.ok) {
      console.error("Wait failed:", res.error);
      process.exit(1);
    }

    const m = res.message;
    console.log("\n================================================================================");
    console.log("🔔 INCOMING INTERCOM MESSAGE FROM PI AGENT");
    console.log("================================================================================");
    console.log(`From:         ${m.from.name} (${m.from.id})`);
    console.log(`From CWD:     ${m.from.cwd}`);
    console.log(`From Model:   ${m.from.model}`);
    console.log(`Message ID:   ${m.id}`);
    if (m.replyTo) {
      console.log(`Reply To:     ${m.replyTo}`);
    }
    if (m.expectsReply) {
      console.log(`Expects Reply: YES (Sender is waiting on this ask!)`);
    }
    console.log("--------------------------------------------------------------------------------");
    console.log(m.content);
    console.log("================================================================================\n");
    return;
  }

  if (cmd === "inbox") {
    await ensureBridgeRunning();
    const res = await sendBridgeRequest({ action: "inbox" });
    console.log(JSON.stringify(res, null, 2));
    return;
  }

  console.log(`
Antigravity <-> Pi Intercom Bridge

Usage:
  agy-intercom start                  Start the background bridge daemon
  agy-intercom stop                   Stop the bridge daemon
  agy-intercom status                 Show connection status to broker
  agy-intercom list                   List all active sessions on the Pi broker
  agy-intercom send <to> <msg> [...]  Send a message to a session (--reply-to <id>, --ask)
  agy-intercom wait-message           Wait for the next message and print it (triggers auto-wakeup)
  agy-intercom inbox                  Inspect all received messages
`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}
