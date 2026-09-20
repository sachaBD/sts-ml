#!/usr/bin/env node

/**
 * Antigravity Model Context Protocol (MCP) Server for Pi Intercom
 * Exposes intercom tools to Antigravity.
 */

const net = require("net");
const path = require("path");
const os = require("os");
const readline = require("readline");

const RUNTIME_DIR = path.join(os.homedir(), ".local/share/antigravity-intercom");
const BRIDGE_SOCK = path.join(RUNTIME_DIR, "bridge.sock");

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

const TOOLS = [
  {
    name: "intercom_list",
    description: "List all active peer agent sessions currently connected to Pi Intercom broker.",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "intercom_send",
    description: "Send a message to another agent on the Pi Intercom broker (e.g. terra-impl, sol-coord).",
    inputSchema: {
      type: "object",
      properties: {
        to: {
          type: "string",
          description: "Recipient agent name or session ID (e.g. 'terra-impl', 'sol-coord')"
        },
        message: {
          type: "string",
          description: "The text message content to send"
        },
        replyTo: {
          type: "string",
          description: "Optional message ID being replied to"
        },
        expectsReply: {
          type: "boolean",
          description: "Whether the sender expects an answer back"
        }
      },
      required: ["to", "message"]
    }
  },
  {
    name: "intercom_status",
    description: "Check the status of the Pi Intercom bridge and inbox.",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "intercom_inbox",
    description: "Inspect received intercom messages and unread queue.",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  }
];

async function handleToolCall(name, args) {
  if (name === "intercom_list") {
    const res = await sendBridgeRequest({ action: "list" });
    if (!res.ok) throw new Error(res.error);
    return JSON.stringify(res.sessions, null, 2);
  }

  if (name === "intercom_send") {
    const res = await sendBridgeRequest({
      action: "send",
      to: args.to,
      text: args.message,
      replyTo: args.replyTo,
      expectsReply: Boolean(args.expectsReply)
    });
    if (!res.ok) throw new Error(res.error);
    return `Message successfully sent to ${args.to}. (Result: ${JSON.stringify(res.result)})`;
  }

  if (name === "intercom_status") {
    const res = await sendBridgeRequest({ action: "status" });
    return JSON.stringify(res, null, 2);
  }

  if (name === "intercom_inbox") {
    const res = await sendBridgeRequest({ action: "inbox" });
    return JSON.stringify(res, null, 2);
  }

  throw new Error(`Unknown tool: ${name}`);
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on("line", async (line) => {
  if (!line.trim()) return;
  try {
    const req = JSON.parse(line);
    const id = req.id;

    if (req.method === "initialize") {
      const resp = {
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: "2024-11-05",
          capabilities: { tools: {} },
          serverInfo: { name: "pi-intercom-mcp", version: "1.0.0" }
        }
      };
      process.stdout.write(JSON.stringify(resp) + "\n");
    } else if (req.method === "notifications/initialized") {
      // No response needed for notification
    } else if (req.method === "tools/list") {
      const resp = {
        jsonrpc: "2.0",
        id,
        result: { tools: TOOLS }
      };
      process.stdout.write(JSON.stringify(resp) + "\n");
    } else if (req.method === "tools/call") {
      const toolName = req.params?.name;
      const toolArgs = req.params?.arguments || {};
      try {
        const textResult = await handleToolCall(toolName, toolArgs);
        const resp = {
          jsonrpc: "2.0",
          id,
          result: {
            content: [{ type: "text", text: textResult }]
          }
        };
        process.stdout.write(JSON.stringify(resp) + "\n");
      } catch (toolErr) {
        const resp = {
          jsonrpc: "2.0",
          id,
          result: {
            isError: true,
            content: [{ type: "text", text: `Tool error: ${toolErr.message}` }]
          }
        };
        process.stdout.write(JSON.stringify(resp) + "\n");
      }
    } else if (req.method === "ping") {
      process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, result: {} }) + "\n");
    } else {
      if (id !== undefined) {
        process.stdout.write(JSON.stringify({
          jsonrpc: "2.0",
          id,
          error: { code: -32601, message: "Method not found" }
        }) + "\n");
      }
    }
  } catch (err) {
    console.error("MCP Protocol error:", err);
  }
});
