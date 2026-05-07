/**
 * research-crew browser frontend — vanilla JS, no framework.
 *
 * UX states:
 *   idle        — initial; form enabled, no run in flight
 *   submitting  — POST /research in progress; submit disabled
 *   streaming   — WebSocket open, agent cards updating live
 *   complete    — run reached terminal state; report rendered
 *   error       — network or API error; message shown, form re-enabled
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const form          = document.getElementById("research-form");
const questionInput = document.getElementById("question-input");
const submitBtn     = document.getElementById("submit-btn");
const cancelBtn     = document.getElementById("cancel-btn");
const formError     = document.getElementById("form-error");
const statusBar     = document.getElementById("status-bar");
const statusLabel   = document.getElementById("status-label");
const runIdDisplay  = document.getElementById("run-id-display");
const agentSection  = document.getElementById("agent-cards-section");
const reportSection = document.getElementById("report-section");
const finalReport   = document.getElementById("final-report");
const errorSection  = document.getElementById("error-section");
const errorMessage  = document.getElementById("error-message");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {WebSocket | null} */
let _ws = null;
/** @type {string | null} */
let _currentRunId = null;

const AGENT_KEYS = ["web_search", "scholar", "code", "news", "wikipedia"];

/** Per-agent local counters (step count, start time). */
const _agentState = {};
AGENT_KEYS.forEach(k => { _agentState[k] = { steps: 0, startedAt: null }; });

// ---------------------------------------------------------------------------
// UI state transitions
// ---------------------------------------------------------------------------

function _setUiIdle() {
  submitBtn.disabled = false;
  cancelBtn.hidden = true;
  _showFormError(null);
}

function _setUiSubmitting() {
  submitBtn.disabled = true;
  cancelBtn.hidden = true;
  _showFormError(null);
  _hideSection(errorSection);
  _hideSection(reportSection);
  _hideSection(agentSection);
  _hideSection(statusBar);
  _resetAgentCards();
}

function _setUiStreaming(runId) {
  submitBtn.disabled = true;
  cancelBtn.hidden = false;
  statusBar.hidden = false;
  statusLabel.textContent = "Streaming…";
  runIdDisplay.textContent = runId;
  agentSection.hidden = false;
}

function _setUiComplete() {
  submitBtn.disabled = false;
  cancelBtn.hidden = true;
  statusLabel.textContent = "Complete";
}

function _setUiError(msg) {
  submitBtn.disabled = false;
  cancelBtn.hidden = true;
  _showFormError(msg);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _showFormError(msg) {
  if (msg) {
    formError.textContent = msg;
    formError.hidden = false;
  } else {
    formError.hidden = true;
    formError.textContent = "";
  }
}

function _hideSection(el) {
  el.hidden = true;
}

function _resetAgentCards() {
  AGENT_KEYS.forEach(agent => {
    _agentState[agent].steps = 0;
    _agentState[agent].startedAt = null;
    const card = document.getElementById("card-" + agent);
    if (!card) return;
    card.className = "agent-card";
    const badge = card.querySelector(".agent-status");
    badge.className = "agent-status badge badge-pending";
    badge.textContent = "pending";
    card.querySelector(".step-count").textContent = "0 steps";
    card.querySelector(".latency").textContent = "";
  });
}

function _updateAgentCard(agent, status, stepCount, latencyText) {
  const card = document.getElementById("card-" + agent);
  if (!card) return;
  // Card border/background
  card.className = "agent-card state-" + status;
  // Badge
  const badge = card.querySelector(".agent-status");
  badge.className = "agent-status badge badge-" + status;
  badge.textContent = status;
  // Meta
  const steps = card.querySelector(".step-count");
  steps.textContent = stepCount + (stepCount === 1 ? " step" : " steps");
  const latency = card.querySelector(".latency");
  latency.textContent = latencyText || "";
}

function _msToDisplay(ms) {
  if (ms == null) return "";
  if (ms < 1000) return ms.toFixed(0) + " ms";
  return (ms / 1000).toFixed(1) + " s";
}

// ---------------------------------------------------------------------------
// WebSocket streaming
// ---------------------------------------------------------------------------

function _openWebSocket(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = proto + "://" + location.host + "/runs/" + runId + "/stream";
  const ws = new WebSocket(url);
  _ws = ws;

  ws.addEventListener("open", () => {
    _setUiStreaming(runId);
  });

  ws.addEventListener("message", (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    _handleWsMessage(msg);
  });

  ws.addEventListener("close", (evt) => {
    _ws = null;
    // If we didn't reach complete/error from a done message, surface it.
    if (evt.code !== 1000) {
      _setUiError("Connection closed unexpectedly (code " + evt.code + ").");
    }
  });

  ws.addEventListener("error", () => {
    _ws = null;
    _setUiError("WebSocket error. Is the API running?");
  });
}

/**
 * Handle a single parsed WebSocket message.
 * Types: snapshot | step | heartbeat | done
 */
function _handleWsMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      _applySnapshot(msg);
      break;
    case "step":
      _applyStep(msg);
      break;
    case "heartbeat":
      statusLabel.textContent = "Heartbeat…";
      setTimeout(() => { statusLabel.textContent = "Streaming…"; }, 1500);
      break;
    case "done":
      _applyDone(msg);
      break;
    default:
      break;
  }
}

/** Apply the initial snapshot (existing steps may already be in the store). */
function _applySnapshot(snap) {
  // Hydrate any already-recorded steps.
  if (Array.isArray(snap.steps)) {
    snap.steps.forEach(step => _applyStep({ ...step, type: "step" }));
  }
  // If already terminal (connected to a finished run), render immediately.
  if (snap.state === "succeeded" || snap.state === "failed") {
    _finalizeRun(snap);
  }
}

function _applyStep(step) {
  const agent = step.agent;
  if (!agent || !_agentState[agent]) return;

  const state = _agentState[agent];
  const status = step.status;

  // Running starts the clock.
  if (status === "running" && !state.startedAt) {
    state.startedAt = Date.now();
  }
  if (status === "running" || status === "pending") {
    // Don't increment for a running→running update.
  } else {
    state.steps += 1;
  }

  let latencyText = "";
  if ((status === "succeeded" || status === "failed" || status === "cached") && state.startedAt) {
    latencyText = _msToDisplay(Date.now() - state.startedAt);
  }

  _updateAgentCard(agent, status, state.steps, latencyText);
}

function _applyDone(_msg) {
  // Wait for the WS close to arrive naturally; _finalizeRun is called
  // from snapshot when state is already terminal, or we poll /runs/{id}.
  statusLabel.textContent = "Finalizing…";
  // Fetch the final run state to render the report.
  fetch("/runs/" + _currentRunId)
    .then(r => r.json())
    .then(run => _finalizeRun(run))
    .catch(() => {
      _setUiError("Run completed but could not fetch final report.");
      _setUiComplete();
    });
}

function _finalizeRun(run) {
  _setUiComplete();
  if (run.state === "succeeded" && run.report) {
    const r = run.report;
    let text = r.summary || "";
    if (Array.isArray(r.citations) && r.citations.length > 0) {
      text += "\n\n## Citations\n";
      r.citations.forEach((c, i) => {
        text += "\n[" + (i + 1) + "] " + c.title + "\n    " + c.url;
        if (c.snippet) text += "\n    " + c.snippet;
      });
    }
    finalReport.textContent = text;
    reportSection.hidden = false;
    _hideSection(errorSection);
  } else if (run.state === "failed") {
    errorMessage.textContent = run.error || "Unknown error.";
    errorSection.hidden = false;
    _hideSection(reportSection);
  }
}

// ---------------------------------------------------------------------------
// Form submit → POST /research
// ---------------------------------------------------------------------------

form.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    _showFormError("Please enter a question.");
    return;
  }

  _setUiSubmitting();

  let runId;
  try {
    const resp = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      _setUiError(
        "API error " + resp.status + ": " + (body.detail || resp.statusText)
      );
      return;
    }
    const data = await resp.json();
    runId = data.run_id;
  } catch (err) {
    _setUiError("Network error: " + err.message);
    return;
  }

  _currentRunId = runId;
  statusBar.hidden = false;
  statusLabel.textContent = "Connecting…";
  runIdDisplay.textContent = runId;
  agentSection.hidden = false;
  _openWebSocket(runId);
});

// ---------------------------------------------------------------------------
// Cancel button
// ---------------------------------------------------------------------------

cancelBtn.addEventListener("click", async () => {
  if (_ws) {
    _ws.close(1000, "user cancelled");
    _ws = null;
  }
  if (_currentRunId) {
    // Best-effort cancel — the API may not have a cancel endpoint, ignore errors.
    try {
      await fetch("/runs/" + _currentRunId + "/cancel", { method: "POST" });
    } catch { /* no-op */ }
  }
  statusLabel.textContent = "Cancelled";
  _setUiIdle();
});
