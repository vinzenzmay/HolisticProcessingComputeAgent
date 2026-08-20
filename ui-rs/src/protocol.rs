//! The wire, in Rust — the twin of `hpca/protocol.py`.
//!
//! The Python side is the authority: every model there is `extra="forbid"`, so
//! a command this file sends must carry exactly the fields the Python class
//! declares — no more, and none renamed. That is why the command structs below
//! serialise their `None`s instead of skipping them: `forced_skill: null` is a
//! declared field and passes, whereas an absent one is only accepted because it
//! has a default, and relying on that would make the two definitions drift
//! apart silently. Events go the other way and are read leniently: an unknown
//! key from a newer core costs that field, not the frame.
//!
//! Framing is NDJSON, and `transport.rs` owns it. This module only says what a
//! frame means.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Asserted on connect; mirrors `protocol.PROTOCOL_VERSION`.
pub const PROTOCOL_VERSION: i64 = 1;

/// The one frame shape, in both directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    #[serde(default)]
    pub seq: i64,
    #[serde(default)]
    pub id: Option<String>,
    /// `type` is a Rust keyword; the wire name is what matters.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub payload: Value,
}

// ------------------------------------------------------------ payload shapes

/// One ordered piece of a turn's working: reasoning, or one tool exchange.
///
/// The call and its result travel together because the UI draws them as one
/// row: `done` is what says whether the second half has landed, and a call
/// still in flight arrives again — filled in — rather than a second part
/// appearing under the first.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Part {
    pub kind: String, // reasoning | call | step
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub tool: String,
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub result: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub failed: bool,
}

impl Part {
    /// Short header for the collapsed row — the port of `transcript.Step.label`.
    pub fn label(&self) -> String {
        if self.kind == "reasoning" {
            return self.kind.clone();
        }
        let mut name = if self.tool.is_empty() {
            "tool".to_string()
        } else {
            self.tool.clone()
        };
        if !self.target.is_empty() {
            name.push_str(" · ");
            name.push_str(&self.target);
        }
        if self.failed {
            name.push_str(" (error)");
        } else if self.kind == "call" && !self.done {
            name.push_str(" …");
        }
        name
    }

    /// Everything under the header — the port of `transcript.Step.body`.
    ///
    /// A call still in flight shows only itself: an empty "result" heading
    /// would read as a tool that answered with nothing.
    pub fn body(&self) -> String {
        let call = self.text.trim();
        let result = self.result.trim();
        if self.kind != "call" {
            return call.to_string();
        }
        if !self.done || result.is_empty() {
            return call.to_string();
        }
        if call.is_empty() {
            return result.to_string();
        }
        format!("{call}\n\n{RESULT_RULE}\n{result}")
    }
}

/// Mirrors `transcript.RESULT_RULE` — the divider between a call and its result.
pub const RESULT_RULE: &str = "── result ──";

/// One rendered line of chat.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Entry {
    pub kind: String, // user | assistant | thinking | error | event | recall
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub steps: i64,
    #[serde(default)]
    pub reasoning_chars: i64,
    #[serde(default)]
    pub parts: Vec<Part>,
    #[serde(default = "minus_one")]
    pub index: i64,
}

fn minus_one() -> i64 {
    -1
}

impl Entry {
    /// One-line gist for the collapsed thinking box — only what is there.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.reasoning_chars > 0 {
            parts.push(format!(
                "{} chars reasoning",
                thousands(self.reasoning_chars)
            ));
        }
        if self.steps > 0 {
            parts.push(format!(
                "{} step{}",
                self.steps,
                if self.steps == 1 { "" } else { "s" }
            ));
        }
        if parts.is_empty() {
            "working".to_string()
        } else {
            parts.join(" · ")
        }
    }
}

/// Python's `f"{n:,}"`. Written out because the chat header shows it on every
/// thinking box, and a locale crate for one comma is not worth the dependency.
pub fn thousands(n: i64) -> String {
    let neg = n < 0;
    let digits = n.abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    if neg {
        out.push('-');
    }
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One line of the sidebar.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionRow {
    pub session_id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub profile: String,
    #[serde(default)]
    pub mode: String,
    /// Open-ended by design: the UI ignores a flag it cannot draw.
    #[serde(default)]
    pub flags: Vec<String>,
}

/// The only panel kind, mirroring `protocol.PANEL_WATCH`.
pub const PANEL_WATCH: &str = "watch";

/// One row of the right column.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PanelRow {
    pub key: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub classes: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub r#ref: String,
}

/// A memory the agent suggests keeping.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Proposal {
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub text: String,
}

// ------------------------------------------------------------------- events

/// Core → UI. Only the variants this prototype draws are typed; everything
/// else is kept as `Other` so an unhandled frame is visibly ignored rather
/// than mistaken for a protocol error.
#[derive(Debug, Clone)]
pub enum Event {
    Hello {
        version: i64,
        profile: String,
        settings_digest: String,
    },
    SessionRows(Vec<SessionRow>),
    ChatReset {
        session_id: String,
        entries: Vec<Entry>,
    },
    ChatAppend {
        session_id: String,
        entry: Entry,
    },
    TurnStarted {
        session_id: String,
    },
    TurnActivity {
        session_id: String,
        activity: String,
        started_at: String,
    },
    TurnUsage {
        session_id: String,
        prompt_tokens: i64,
        max_model_len: Option<i64>,
    },
    TurnFinished {
        session_id: String,
        /// The answer. Redundant once the core emits `chat.append` for the
        /// assistant entry — which it does not yet — so a front-end has to be
        /// able to render a turn from this alone. See `apply_event`.
        reply: Option<String>,
    },
    TurnFailed {
        session_id: String,
        error: String,
    },
    DecisionRequested {
        session_id: String,
        payload: Value,
    },
    DecisionCleared {
        session_id: String,
    },
    PanelUpdate {
        rows: Vec<PanelRow>,
    },
    MemoryProposals {
        session_id: String,
        proposals: Vec<Proposal>,
    },
    ConfirmRequested {
        id: String,
        question: String,
    },
    ContextEstimate {
        session_id: String,
        used: i64,
        window: i64,
    },
    Notify {
        severity: String,
        text: String,
        timeout: Option<f64>,
    },
    /// A frame whose type this front-end does not draw.
    Other(String),
}

/// What went wrong reading a frame. One error for the module, so a reader loop
/// needs one `match` arm and can carry on rather than dropping the connection —
/// the same call the Python `ProtocolError` makes.
#[derive(Debug)]
pub struct ProtocolError(pub String);

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ProtocolError {}

/// Read one field out of a payload, defaulting when it is absent.
fn field<T: serde::de::DeserializeOwned + Default>(payload: &Value, name: &str) -> T {
    payload
        .get(name)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

fn opt_field<T: serde::de::DeserializeOwned>(payload: &Value, name: &str) -> Option<T> {
    payload
        .get(name)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

impl Event {
    /// The typed body of an envelope, or `Other` if this front-end has no
    /// drawing for it.
    pub fn from_envelope(env: &Envelope) -> Result<Event, ProtocolError> {
        let p = &env.payload;
        let ev = match env.kind.as_str() {
            "hello" => Event::Hello {
                version: opt_field(p, "version").unwrap_or(PROTOCOL_VERSION),
                profile: field(p, "profile"),
                settings_digest: field(p, "settings_digest"),
            },
            "session.rows" => Event::SessionRows(field(p, "rows")),
            "chat.reset" => Event::ChatReset {
                session_id: field(p, "session_id"),
                entries: field(p, "entries"),
            },
            "chat.append" => Event::ChatAppend {
                session_id: field(p, "session_id"),
                entry: opt_field(p, "entry")
                    .ok_or_else(|| ProtocolError("chat.append without an entry".into()))?,
            },
            "turn.started" => Event::TurnStarted {
                session_id: field(p, "session_id"),
            },
            "turn.activity" => Event::TurnActivity {
                session_id: field(p, "session_id"),
                activity: field(p, "activity"),
                started_at: field(p, "started_at"),
            },
            "turn.usage" => Event::TurnUsage {
                session_id: field(p, "session_id"),
                prompt_tokens: field(p, "prompt_tokens"),
                max_model_len: opt_field(p, "max_model_len"),
            },
            "turn.finished" => Event::TurnFinished {
                session_id: field(p, "session_id"),
                reply: opt_field(p, "reply"),
            },
            "turn.failed" => Event::TurnFailed {
                session_id: field(p, "session_id"),
                error: field(p, "error"),
            },
            "decision.requested" => Event::DecisionRequested {
                session_id: field(p, "session_id"),
                payload: p.get("payload").cloned().unwrap_or(Value::Null),
            },
            "decision.cleared" => Event::DecisionCleared {
                session_id: field(p, "session_id"),
            },
            "panel.update" => Event::PanelUpdate {
                rows: field(p, "rows"),
            },
            "memory.proposals" => Event::MemoryProposals {
                session_id: field(p, "session_id"),
                proposals: field(p, "proposals"),
            },
            "confirm.requested" => Event::ConfirmRequested {
                id: field(p, "id"),
                question: field(p, "question"),
            },
            "context.estimate" => Event::ContextEstimate {
                session_id: field(p, "session_id"),
                used: field(p, "used"),
                window: field(p, "window"),
            },
            "notify" => Event::Notify {
                severity: opt_field(p, "severity").unwrap_or_else(|| "information".into()),
                text: field(p, "text"),
                timeout: opt_field(p, "timeout"),
            },
            other => Event::Other(other.to_string()),
        };
        Ok(ev)
    }
}

// ----------------------------------------------------------------- commands

/// UI → core. Each variant serialises to the payload its Python twin declares.
#[derive(Debug, Clone)]
pub enum Command {
    SessionList,
    SessionNew { profile: String, backend: Option<String> },
    SessionOpen { session_id: String },
    SessionClose,
    SessionRename { session_id: String, title: String },
    SessionRetitle { session_id: String },
    SessionDelete { session_id: String },
    SessionFocus { session_id: Option<String> },
    TurnSubmit { session_id: String, text: String, forced_skill: Option<String> },
    TurnInterrupt { session_id: String },
    DecisionResolve { session_id: String, approved: bool, reason: String },
    CommandRun { name: String, args: String, session_id: Option<String> },
    ConfirmResolve { id: String, confirmed: bool },
    MemoryResolve { session_id: String, approved: Vec<bool> },
    ModeSet { session_id: String, mode: String },
    ThinkingSet { session_id: String, effort: String },
    ProcessKill { pid: i64 },
    WatchDrop { watch_id: i64 },
    Shutdown,
}

impl Command {
    /// The envelope type string, which both ends dispatch on.
    pub fn type_name(&self) -> &'static str {
        match self {
            Command::SessionList => "session.list",
            Command::SessionNew { .. } => "session.new",
            Command::SessionOpen { .. } => "session.open",
            Command::SessionClose => "session.close",
            Command::SessionRename { .. } => "session.rename",
            Command::SessionRetitle { .. } => "session.retitle",
            Command::SessionDelete { .. } => "session.delete",
            Command::SessionFocus { .. } => "session.focus",
            Command::TurnSubmit { .. } => "turn.submit",
            Command::TurnInterrupt { .. } => "turn.interrupt",
            Command::DecisionResolve { .. } => "decision.resolve",
            Command::CommandRun { .. } => "command.run",
            Command::ConfirmResolve { .. } => "confirm.resolve",
            Command::MemoryResolve { .. } => "memory.resolve",
            Command::ModeSet { .. } => "mode.set",
            Command::ThinkingSet { .. } => "thinking.set",
            Command::ProcessKill { .. } => "process.kill",
            Command::WatchDrop { .. } => "watch.drop",
            Command::Shutdown => "shutdown",
        }
    }

    /// The payload, with every declared field present — see the module note on
    /// why `None` is written out rather than skipped.
    pub fn payload(&self) -> Value {
        use serde_json::json;
        match self {
            Command::SessionList | Command::SessionClose | Command::Shutdown => json!({}),
            Command::SessionNew { profile, backend } => {
                json!({ "profile": profile, "backend": backend })
            }
            Command::SessionOpen { session_id } => json!({ "session_id": session_id }),
            Command::SessionRename { session_id, title } => {
                json!({ "session_id": session_id, "title": title })
            }
            Command::SessionRetitle { session_id } => json!({ "session_id": session_id }),
            Command::SessionDelete { session_id } => json!({ "session_id": session_id }),
            Command::SessionFocus { session_id } => json!({ "session_id": session_id }),
            Command::TurnSubmit { session_id, text, forced_skill } => {
                json!({ "session_id": session_id, "text": text, "forced_skill": forced_skill })
            }
            Command::TurnInterrupt { session_id } => json!({ "session_id": session_id }),
            Command::DecisionResolve { session_id, approved, reason } => {
                json!({ "session_id": session_id, "approved": approved, "reason": reason })
            }
            Command::CommandRun { name, args, session_id } => {
                json!({ "name": name, "args": args, "session_id": session_id })
            }
            Command::ConfirmResolve { id, confirmed } => {
                json!({ "id": id, "confirmed": confirmed })
            }
            Command::MemoryResolve { session_id, approved } => {
                json!({ "session_id": session_id, "approved": approved })
            }
            Command::ModeSet { session_id, mode } => {
                json!({ "session_id": session_id, "mode": mode })
            }
            Command::ThinkingSet { session_id, effort } => {
                json!({ "session_id": session_id, "effort": effort })
            }
            Command::ProcessKill { pid } => json!({ "pid": pid }),
            Command::WatchDrop { watch_id } => json!({ "watch_id": watch_id }),
        }
    }

    /// The frame to put on the wire. `seq` belongs to the connection, not the
    /// caller, so `transport` fills it in.
    pub fn to_envelope(&self, seq: i64, id: Option<String>) -> Envelope {
        Envelope {
            seq,
            id,
            kind: self.type_name().to_string(),
            payload: self.payload(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_matches_pythons_comma_format() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(-4321), "-4,321");
    }

    #[test]
    fn a_call_awaiting_its_result_shows_only_the_call() {
        let part = Part {
            kind: "call".into(),
            text: "run_bash(...)".into(),
            tool: "run_bash".into(),
            done: false,
            ..Default::default()
        };
        assert_eq!(part.label(), "run_bash …");
        assert_eq!(part.body(), "run_bash(...)");
    }

    #[test]
    fn a_finished_call_puts_the_result_under_the_rule() {
        let part = Part {
            kind: "call".into(),
            text: "cat f".into(),
            tool: "run_bash".into(),
            target: "f".into(),
            result: "hello".into(),
            done: true,
            ..Default::default()
        };
        assert_eq!(part.label(), "run_bash · f");
        assert!(part.body().contains(RESULT_RULE));
        assert!(part.body().ends_with("hello"));
    }

    #[test]
    fn a_failed_call_says_so_on_the_collapsed_row() {
        let part = Part {
            kind: "call".into(),
            tool: "edit_file".into(),
            failed: true,
            done: true,
            ..Default::default()
        };
        assert_eq!(part.label(), "edit_file (error)");
    }

    #[test]
    fn summary_reports_only_what_is_there() {
        let mut e = Entry { kind: "thinking".into(), ..Default::default() };
        assert_eq!(e.summary(), "working");
        e.steps = 1;
        assert_eq!(e.summary(), "1 step");
        e.reasoning_chars = 2500;
        assert_eq!(e.summary(), "2,500 chars reasoning · 1 step");
    }

    #[test]
    fn commands_carry_every_declared_field() {
        // extra="forbid" on the far side means a missing declared field is the
        // failure this asserts against, not a stylistic preference.
        let c = Command::TurnSubmit {
            session_id: "s1".into(),
            text: "hi".into(),
            forced_skill: None,
        };
        let p = c.payload();
        assert!(p.get("session_id").is_some());
        assert!(p.get("text").is_some());
        assert!(p.get("forced_skill").is_some(), "null must be written, not skipped");
    }

    #[test]
    fn an_unknown_event_type_is_ignored_rather_than_fatal() {
        let env = Envelope {
            seq: 1,
            id: None,
            kind: "something.new".into(),
            payload: serde_json::json!({}),
        };
        match Event::from_envelope(&env).unwrap() {
            Event::Other(t) => assert_eq!(t, "something.new"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
