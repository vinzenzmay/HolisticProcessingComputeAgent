//! A ratatui front-end for HPCA, talking the core protocol.
//!
//! It is a *replacement for the UI process only*. `specs-core-process.md`
//! already draws the seam this plugs into — "the protocol is designed so a
//! second front-end is a later, additive front-end, not a redesign" — so no
//! part of the agent runtime is ported here, and none is imported. What crosses
//! the socket is what `hpca/protocol.py` says crosses it.
//!
//! Three ways to run it:
//!
//!   hpca-ui --socket <path>   attach to a core and draw
//!   hpca-ui --snapshot [n]    render one frame off-screen and print it
//!   hpca-ui --bench [n]       time frames against a synthetic transcript
//!
//! The last two exist because the question this prototype answers is partly a
//! performance one, and "it feels smoother" is not an answer.

mod app;
mod bench;
mod input;
mod protocol;
mod transport;
mod ui;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::app::{App, ChatFocus, ChatLayout, Column, Decision, Working};
use crate::protocol::{Command, Event};
use crate::transport::Incoming;

/// The spinner's period, from `WorkingIndicator.INTERVAL`.
const TICK: Duration = Duration::from_millis(80);

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut socket: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" => {
                socket = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--snapshot" => {
                let n = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(12);
                bench::snapshot(n);
                return Ok(());
            }
            "--bench" => {
                let n = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(300);
                bench::bench(n);
                return Ok(());
            }
            "--probe" => {
                let path = args
                    .get(i + 1)
                    .map(PathBuf::from)
                    .expect("--probe needs a socket path");
                return probe(path).await;
            }
            "--help" | "-h" => {
                eprintln!("hpca-ui --socket <path> | --snapshot [entries] | --bench [messages]");
                return Ok(());
            }
            _ => i += 1,
        }
    }

    let Some(socket) = socket else {
        eprintln!("no --socket given; try --snapshot to see a frame, or --help");
        std::process::exit(2);
    };
    run(socket).await
}

async fn run(socket: PathBuf) -> Result<()> {
    let conn = transport::connect_unix(&socket).await?;
    let mut app = App::default();
    let mut layout = ChatLayout::default();

    enable_raw_mode()?;
    let mut out = std::io::stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(out))?;

    let result = event_loop(&mut term, &mut app, &mut layout, conn).await;

    disable_raw_mode()?;
    execute!(term.backend_mut(), LeaveAlternateScreen)?;
    term.show_cursor()?;
    result
}

async fn event_loop(
    term: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    layout: &mut ChatLayout,
    mut conn: transport::Connection,
) -> Result<()> {
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut tick: u64 = 0;

    conn.send(Command::SessionList);

    loop {
        term.draw(|f| ui::draw(f, app, layout, tick))?;
        app.frames += 1;
        if app.should_quit {
            conn.send(Command::SessionFocus { session_id: None });
            return Ok(());
        }

        tokio::select! {
            // A tick only advances the spinner. It redraws — one line changes —
            // but touches no state, so the layout cache survives it.
            _ = ticker.tick() => {
                tick = tick.wrapping_add(1);
                app.expire_toasts();
            }
            Some(Ok(ev)) = keys.next() => {
                match ev {
                    TermEvent::Key(k) if k.kind == KeyEventKind::Press => {
                        for cmd in input::handle_key(app, layout, k) {
                            conn.send(cmd);
                        }
                    }
                    TermEvent::Resize(_, _) => { /* the cache keys on width */ }
                    _ => {}
                }
            }
            Some(msg) = conn.events.recv() => {
                match msg {
                    Incoming::Event(e) => apply_event(app, e),
                    // Visible rather than swallowed: a mismatch between the two
                    // protocol definitions must not look like a quiet core.
                    Incoming::Bad(why) => app.notify("error", &format!("bad frame: {why}"), None),
                    Incoming::Closed => {
                        app.connected = false;
                        app.status = "the core closed the connection".into();
                        app.notify("error", "the core closed the connection", Some(30.0));
                    }
                }
            }
        }
    }
}

/// Fold one event into the state. The only place the wire touches the UI.
pub fn apply_event(app: &mut App, ev: Event) {
    match ev {
        Event::Hello { version, profile, .. } => {
            app.connected = true;
            app.profile = profile;
            app.status = if version == protocol::PROTOCOL_VERSION {
                "connected".into()
            } else {
                format!("protocol {version}, expected {}", protocol::PROTOCOL_VERSION)
            };
        }
        Event::SessionRows(rows) => {
            app.sessions = rows;
            if app.session_cursor >= app.sessions.len() {
                app.session_cursor = app.sessions.len().saturating_sub(1);
            }
        }
        // The snapshot before the deltas.
        Event::ChatReset { session_id, entries } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.entries = entries;
                // An expansion belongs to the entries it was opened on; fresh
                // ones are not those.
                app.expansion.clear();
                app.chat_cursor = None;
                app.touch();
            }
        }
        // Why the history stops crossing the socket every turn.
        Event::ChatAppend { session_id, entry } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                if entry.kind == "assistant" {
                    app.appended_this_turn = true;
                }
                app.entries.push(entry);
                app.touch();
            }
        }
        Event::TurnStarted { session_id } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.working = Some(Working {
                    activity: "working".into(),
                    started: Instant::now(),
                    interruptible: true,
                });
                app.appended_this_turn = false;
                app.touch();
            }
        }
        // One event replaces show_working / hide_working / report_activity.
        Event::TurnActivity { session_id, activity, .. } => {
            if app.active_session.as_deref() != Some(session_id.as_str()) {
                return;
            }
            if activity.is_empty() {
                app.working = None;
                app.touch();
            } else if let Some(w) = &mut app.working {
                // The clock belongs to the turn, not to the step: the question
                // the user has while waiting is "how long since I asked?".
                w.activity = activity;
            } else {
                app.working = Some(Working {
                    activity,
                    started: Instant::now(),
                    interruptible: true,
                });
                app.touch();
            }
        }
        Event::TurnUsage { session_id, prompt_tokens, max_model_len } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.context_used = prompt_tokens;
                app.context_estimated = false;
                if let Some(w) = max_model_len {
                    app.context_window = w;
                }
            }
        }
        Event::TurnFinished { session_id, reply } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.working = None;
                // The core does not emit `chat.append` yet, so the reply on
                // this event is the only copy of the answer. Once it does, the
                // entry will already be here and this must not duplicate it —
                // hence the check rather than an unconditional push.
                if let Some(reply) = reply {
                    if !reply.trim().is_empty() && !app.appended_this_turn {
                        app.entries.push(crate::protocol::Entry {
                            kind: "assistant".into(),
                            text: reply,
                            ..Default::default()
                        });
                    }
                }
                app.appended_this_turn = false;
                app.touch();
            }
        }
        Event::TurnFailed { session_id, error } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.working = None;
                app.touch();
            }
            // The core also lands this as a chat entry; the toast is what makes
            // it visible when the failure is on a session the user left.
            app.notify("error", &error, Some(15.0));
        }
        Event::DecisionRequested { session_id, payload } => {
            // The graph's interrupt value is passed through untouched, so the
            // front-end reads it defensively rather than assuming a shape.
            let question = payload
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or("approve this call?")
                .to_string();
            let detail = payload
                .get("command")
                .or_else(|| payload.get("script"))
                .or_else(|| payload.get("detail"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            app.decision = Some(Decision { session_id, question, detail, reason: None });
            app.touch();
        }
        Event::DecisionCleared { session_id } => {
            if app.decision.as_ref().map(|d| d.session_id.as_str()) == Some(session_id.as_str()) {
                app.decision = None;
                app.touch();
            }
        }
        // Rows are cheap and diffing them is the UI's job.
        Event::PanelUpdate { rows } => {
            app.panel = rows;
            if app.panel_cursor >= app.panel.len() {
                app.panel_cursor = app.panel.len().saturating_sub(1);
            }
        }
        Event::MemoryProposals { proposals, .. } => {
            app.notify(
                "information",
                &format!("{} memories proposed (/memorize to review)", proposals.len()),
                Some(10.0),
            );
        }
        Event::ConfirmRequested { question, .. } => {
            app.notify("warning", &question, Some(20.0));
        }
        Event::ContextEstimate { session_id, used, window } => {
            if app.active_session.as_deref() == Some(session_id.as_str()) {
                app.context_used = used;
                app.context_window = window;
                app.context_estimated = true;
            }
        }
        Event::Notify { severity, text, timeout } => app.notify(&severity, &text, timeout),
        Event::Other(_) => {}
    }
}

/// Drive a real core headlessly and print what came back.
///
/// This is the interoperability check: it opens a session, sends a turn,
/// answers whatever approval comes back, and then renders a frame off-screen.
/// Every command it sends is parsed by the Python `protocol.parse` on the far
/// side, so a payload this front-end shaped wrongly fails there rather than
/// here.
async fn probe(socket: PathBuf) -> Result<()> {
    let mut conn = transport::connect_unix(&socket).await?;
    let mut app = App::default();
    let mut seen: Vec<String> = Vec::new();

    conn.send(Command::SessionList);

    // A core that cannot yet answer `session.list` (the half-landed one) still
    // has a session store behind it, so the harness can name the session it
    // seeded and the probe will focus it directly.
    let seeded = std::env::var("HPCA_PROBE_SESSION").ok();
    if let Some(seeded) = &seeded {
        conn.send(Command::SessionFocus { session_id: Some(seeded.clone()) });
        app.active_session = Some(seeded.clone());
        // A real generation takes tens of seconds, so the turn goes out now
        // rather than waiting for a `chat.reset` this core cannot send.
        if let Ok(text) = std::env::var("HPCA_PROBE_TURN") {
            conn.send(Command::TurnSubmit {
                session_id: seeded.clone(),
                text,
                forced_skill: None,
            });
        }
    }

    let secs = std::env::var("HPCA_PROBE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut opened = seeded.is_some();
    let mut submitted = seeded.is_some();

    while Instant::now() < deadline {
        let left = deadline.saturating_duration_since(Instant::now());
        let Ok(Some(msg)) = tokio::time::timeout(left, conn.events.recv()).await else {
            break;
        };
        match msg {
            Incoming::Event(e) => {
                seen.push(event_name(&e).to_string());
                // Answer an approval the moment it arrives, so the probe
                // exercises the round trip rather than just the inbound half.
                if let Event::DecisionRequested { session_id, .. } = &e {
                    let sid = session_id.clone();
                    apply_event(&mut app, e);
                    conn.send(Command::DecisionResolve {
                        session_id: sid,
                        approved: true,
                        reason: String::new(),
                    });
                    continue;
                }
                apply_event(&mut app, e);
            }
            Incoming::Bad(why) => {
                println!("BAD FRAME: {why}");
                seen.push(format!("BAD({why})"));
            }
            Incoming::Closed => {
                println!("core closed the connection");
                break;
            }
        }

        if !opened && !app.sessions.is_empty() {
            opened = true;
            let sid = app.sessions[0].session_id.clone();
            app.active_session = Some(sid.clone());
            app.mode = app.sessions[0].mode.clone();
            conn.send(Command::SessionOpen { session_id: sid.clone() });
            conn.send(Command::SessionFocus { session_id: Some(sid) });
        }
        if opened && !submitted && !app.entries.is_empty() {
            submitted = true;
            let sid = app.active_session.clone().unwrap();
            conn.send(Command::TurnSubmit {
                session_id: sid.clone(),
                text: "resubmit it with a longer limit".into(),
                forced_skill: None,
            });
            // A slash line and a mode change too, so more of the command
            // surface is validated by the far side's parser.
            conn.send(Command::CommandRun {
                name: "compact".into(),
                args: String::new(),
                session_id: Some(sid.clone()),
            });
            conn.send(Command::ModeSet { session_id: sid, mode: "auto".into() });
            conn.send(Command::WatchDrop { watch_id: 2 });
        }
    }

    println!("── probe result ─────────────────────────────────────────");
    println!("connected      : {}", app.connected);
    println!("status         : {}", app.status);
    println!("profile        : {}", app.profile);
    println!("sessions       : {}", app.sessions.len());
    println!("watch rows     : {}", app.panel.len());
    println!("chat entries   : {}", app.entries.len());
    println!("context        : {}", ui::context_text(&app));
    println!("bad frames     : {}", seen.iter().filter(|s| s.starts_with("BAD")).count());
    println!("events seen    : {}", seen.join(", "));
    println!("─────────────────────────────────────────────────────────");

    // And a frame, so the state that came off a real wire is shown drawn.
    let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(110, 30))?;
    let mut layout = ChatLayout::default();
    term.draw(|f| ui::draw(f, &mut app, &mut layout, 2))?;
    let buf = term.backend().buffer();
    for y in 0..30u16 {
        let line: String = (0..110u16).map(|x| buf[(x, y)].symbol()).collect();
        println!("{}", line.trim_end());
    }
    Ok(())
}

fn event_name(e: &Event) -> &'static str {
    match e {
        Event::Hello { .. } => "hello",
        Event::SessionRows(_) => "session.rows",
        Event::ChatReset { .. } => "chat.reset",
        Event::ChatAppend { .. } => "chat.append",
        Event::TurnStarted { .. } => "turn.started",
        Event::TurnActivity { .. } => "turn.activity",
        Event::TurnUsage { .. } => "turn.usage",
        Event::TurnFinished { .. } => "turn.finished",
        Event::TurnFailed { .. } => "turn.failed",
        Event::DecisionRequested { .. } => "decision.requested",
        Event::DecisionCleared { .. } => "decision.cleared",
        Event::PanelUpdate { .. } => "panel.update",
        Event::MemoryProposals { .. } => "memory.proposals",
        Event::ConfirmRequested { .. } => "confirm.requested",
        Event::ContextEstimate { .. } => "context.estimate",
        Event::Notify { .. } => "notify",
        Event::Other(_) => "other",
    }
}

/// Used by the bench and snapshot paths, and by tests.
pub fn focus_chat(app: &mut App) {
    app.column = Column::Chat;
    app.chat_focus = ChatFocus::Input;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Entry, SessionRow};

    fn opened() -> App {
        App { active_session: Some("s1".into()), ..Default::default() }
    }

    #[test]
    fn a_reset_for_another_session_is_ignored() {
        let mut app = opened();
        apply_event(
            &mut app,
            Event::ChatReset {
                session_id: "other".into(),
                entries: vec![Entry { kind: "user".into(), ..Default::default() }],
            },
        );
        assert!(app.entries.is_empty(), "a background session must not overwrite the log");
    }

    #[test]
    fn an_append_lands_and_bumps_the_revision() {
        let mut app = opened();
        let before = app.revision;
        apply_event(
            &mut app,
            Event::ChatAppend {
                session_id: "s1".into(),
                entry: Entry { kind: "assistant".into(), text: "hi".into(), ..Default::default() },
            },
        );
        assert_eq!(app.entries.len(), 1);
        assert_ne!(app.revision, before, "the layout cache must learn it is stale");
    }

    #[test]
    fn an_empty_activity_means_the_turn_stopped_working() {
        let mut app = opened();
        apply_event(&mut app, Event::TurnStarted { session_id: "s1".into() });
        assert!(app.working.is_some());
        apply_event(
            &mut app,
            Event::TurnActivity {
                session_id: "s1".into(),
                activity: String::new(),
                started_at: String::new(),
            },
        );
        assert!(app.working.is_none());
    }

    #[test]
    fn naming_the_step_leaves_the_clock_alone() {
        let mut app = opened();
        apply_event(&mut app, Event::TurnStarted { session_id: "s1".into() });
        let started = app.working.as_ref().unwrap().started;
        apply_event(
            &mut app,
            Event::TurnActivity {
                session_id: "s1".into(),
                activity: "run_bash".into(),
                started_at: String::new(),
            },
        );
        let w = app.working.as_ref().unwrap();
        assert_eq!(w.activity, "run_bash");
        assert_eq!(w.started, started, "the count answers 'how long since I asked?'");
    }

    #[test]
    fn a_decision_payload_of_an_unexpected_shape_still_draws() {
        let mut app = opened();
        apply_event(
            &mut app,
            Event::DecisionRequested {
                session_id: "s1".into(),
                payload: serde_json::json!({ "surprise": 1 }),
            },
        );
        let d = app.decision.unwrap();
        assert_eq!(d.question, "approve this call?");
        assert_eq!(d.detail, "");
    }

    #[test]
    fn a_shrinking_sidebar_does_not_strand_the_cursor() {
        let mut app = App::default();
        apply_event(
            &mut app,
            Event::SessionRows(vec![
                SessionRow { session_id: "a".into(), ..Default::default() },
                SessionRow { session_id: "b".into(), ..Default::default() },
            ]),
        );
        app.session_cursor = 1;
        apply_event(
            &mut app,
            Event::SessionRows(vec![SessionRow {
                session_id: "a".into(),
                ..Default::default()
            }]),
        );
        assert_eq!(app.session_cursor, 0);
    }
}
