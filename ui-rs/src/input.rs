//! Keys. The Textual `BINDINGS` tables, one for one.
//!
//! The ordering rules the original states in comments are load-bearing and kept
//! here: (q) quits but must still reach the chat entry as a letter, so it is
//! offered only where no typing happens; and escape is *not* a priority binding
//! — it is reached only when nothing nearer claimed the key, so a prompt that
//! uses escape to mean "leave this" still outranks "stop the agent".

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, ChatFocus, ChatLayout, ChatRow, Column, Expansion};
use crate::protocol::Command;

/// How long the first escape stays armed as half of a stop gesture.
const ESC_PAIR_WINDOW: Duration = Duration::from_secs(1);

/// The modes `shift+tab` cycles through.
const MODES: [&str; 3] = ["manual", "auto", "full-auto"];

pub fn handle_key(app: &mut App, layout: &ChatLayout, key: KeyEvent) -> Vec<Command> {
    let mut out = Vec::new();
    let typing = app.column == Column::Chat
        && app.chat_focus == ChatFocus::Input
        && app.decision.is_none();

    // A refusal being explained owns the keyboard until it is sent or dropped.
    if let Some(d) = &mut app.decision {
        if d.reason.is_some() {
            return decision_reason_key(app, key);
        }
    }

    match (key.code, key.modifiers) {
        // shift+tab is the binding that works everywhere: ctrl+m arrives as
        // Enter in most terminals, so it cannot be told apart from a send.
        (KeyCode::BackTab, _) => {
            if let Some(sid) = app.active_session.clone() {
                let at = MODES.iter().position(|m| *m == app.mode).unwrap_or(0);
                let next = MODES[(at + 1) % MODES.len()];
                app.mode = next.to_string();
                out.push(Command::ModeSet { session_id: sid, mode: next.to_string() });
            }
            return out;
        }
        (KeyCode::Left, m) if !typing || m.contains(KeyModifiers::ALT) => {
            app.column = app.column.shift(-1);
            return out;
        }
        (KeyCode::Right, m) if !typing || m.contains(KeyModifiers::ALT) => {
            app.column = app.column.shift(1);
            return out;
        }
        (KeyCode::Esc, _) => {
            return escape(app, out);
        }
        _ => {}
    }

    if let Some(d) = &app.decision {
        // The approval bar answers first, whatever column has focus.
        let sid = d.session_id.clone();
        match key.code {
            KeyCode::Char('y') => {
                app.decision = None;
                app.touch();
                out.push(Command::DecisionResolve {
                    session_id: sid,
                    approved: true,
                    reason: String::new(),
                });
                return out;
            }
            KeyCode::Char('n') => {
                app.decision = None;
                app.touch();
                out.push(Command::DecisionResolve {
                    session_id: sid,
                    approved: false,
                    reason: String::new(),
                });
                return out;
            }
            KeyCode::Char('r') => {
                if let Some(d) = &mut app.decision {
                    d.reason = Some(String::new());
                }
                return out;
            }
            _ => {}
        }
    }

    match app.column {
        Column::Sessions => sessions_key(app, key, &mut out),
        Column::Watchers => watchers_key(app, key, &mut out),
        Column::Chat => chat_key(app, layout, key, &mut out),
    }
    out
}

/// Deliberately not a priority binding — see the module note.
fn escape(app: &mut App, mut out: Vec<Command>) -> Vec<Command> {
    // Inside the entry, escape leaves the entry rather than stopping anything.
    if app.column == Column::Chat && app.chat_focus == ChatFocus::Input {
        app.chat_focus = ChatFocus::Log;
        app.chat_cursor = None;
        return out;
    }
    let now = Instant::now();
    let armed = app
        .esc_armed_at
        .map(|t| now.duration_since(t) < ESC_PAIR_WINDOW)
        .unwrap_or(false);
    if armed && app.working.is_some() {
        app.esc_armed_at = None;
        if let Some(sid) = app.active_session.clone() {
            app.status = "interrupting…".into();
            out.push(Command::TurnInterrupt { session_id: sid });
        }
    } else {
        app.esc_armed_at = Some(now);
    }
    out
}

fn sessions_key(app: &mut App, key: KeyEvent, out: &mut Vec<Command>) {
    match key.code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Up => {
            app.session_cursor = app.session_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            if app.session_cursor + 1 < app.sessions.len() {
                app.session_cursor += 1;
            }
        }
        KeyCode::Char('n') => {
            out.push(Command::SessionNew {
                profile: app.profile.clone(),
                backend: None,
            });
        }
        KeyCode::Enter => {
            // Read what is needed off the row before touching the app: opening
            // a session rewrites the state the row is borrowed from.
            let picked = app
                .selected_session()
                .map(|r| (r.session_id.clone(), r.mode.clone()));
            if let Some((sid, mode)) = picked {
                app.swap_draft(Some(&sid));
                app.active_session = Some(sid.clone());
                app.mode = mode;
                app.entries.clear();
                app.expansion.clear();
                app.chat_cursor = None;
                app.touch();
                out.push(Command::SessionOpen { session_id: sid.clone() });
                out.push(Command::SessionFocus { session_id: Some(sid) });
                app.column = Column::Chat;
                app.chat_focus = ChatFocus::Input;
            }
        }
        // Renaming applies to a session, never to "(new session)".
        KeyCode::Char('t') => {
            if let Some(row) = app.selected_session() {
                out.push(Command::SessionRetitle { session_id: row.session_id.clone() });
            }
        }
        KeyCode::Char('d') => {
            if let Some(row) = app.selected_session() {
                out.push(Command::SessionDelete { session_id: row.session_id.clone() });
            }
        }
        KeyCode::Char('r') => {
            app.notify("information", "rename: a modal this prototype does not draw", None);
        }
        KeyCode::Char('c') | KeyCode::Char('m') | KeyCode::Char('a') => {
            app.notify("information", "that screen is out of the prototype's scope", None);
        }
        _ => {}
    }
}

fn watchers_key(app: &mut App, key: KeyEvent, out: &mut Vec<Command>) {
    match key.code {
        KeyCode::Up => app.panel_cursor = app.panel_cursor.saturating_sub(1),
        KeyCode::Down => {
            if app.panel_cursor + 1 < app.panel.len() {
                app.panel_cursor += 1;
            }
        }
        KeyCode::Char('d') => {
            // The UI turns `kind` plus `ref` into the command a keypress sends.
            if let Some(row) = app.selected_panel() {
                if row.kind == crate::protocol::PANEL_WATCH {
                    if let Ok(id) = row.r#ref.parse::<i64>() {
                        out.push(Command::WatchDrop { watch_id: id });
                    }
                }
            }
        }
        _ => {}
    }
}

fn chat_key(app: &mut App, layout: &ChatLayout, key: KeyEvent, out: &mut Vec<Command>) {
    match app.chat_focus {
        ChatFocus::Log => chat_log_key(app, layout, key, out),
        ChatFocus::Input => chat_input_key(app, key, out),
    }
}

fn chat_log_key(app: &mut App, layout: &ChatLayout, key: KeyEvent, out: &mut Vec<Command>) {
    match key.code {
        KeyCode::Up => {
            let cur = app.chat_cursor.unwrap_or(layout.rows.len());
            app.chat_cursor = Some(cur.saturating_sub(1));
        }
        KeyCode::Down => {
            let cur = app.chat_cursor.unwrap_or(0);
            if cur + 1 < layout.rows.len() {
                app.chat_cursor = Some(cur + 1);
            }
        }
        KeyCode::Enter => {
            let Some(cur) = app.chat_cursor else {
                app.chat_focus = ChatFocus::Input;
                return;
            };
            match layout.rows.get(cur) {
                // A turn's own line is selectable to abort it, whatever phase
                // it is in.
                Some(ChatRow::Working) => {
                    if let Some(sid) = app.active_session.clone() {
                        out.push(Command::TurnInterrupt { session_id: sid });
                    }
                }
                Some(ChatRow::Entry(i)) => {
                    let i = *i;
                    if app.entries.get(i).map(|e| e.kind == "thinking") == Some(true) {
                        let e = app.expansion.entry(i).or_default();
                        e.open = !e.open;
                        app.touch();
                    }
                }
                Some(ChatRow::Part { entry, part }) => {
                    let (entry, part) = (*entry, *part);
                    let e = app.expansion.entry(entry).or_insert_with(Expansion::default);
                    let open = e.parts.get(&part).copied().unwrap_or(false);
                    e.parts.insert(part, !open);
                    app.touch();
                }
                None => {}
            }
        }
        // Any letter drops back into the entry and types it — the log is for
        // reading, and the user who starts typing means to write a message.
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            if c == 'q' {
                app.should_quit = true;
                return;
            }
            app.chat_focus = ChatFocus::Input;
            app.chat_cursor = None;
            app.input.push(c);
        }
        _ => {}
    }
}

fn chat_input_key(app: &mut App, key: KeyEvent, out: &mut Vec<Command>) {
    match (key.code, key.modifiers) {
        // shift+enter / alt+enter start a new line; plain enter sends.
        (KeyCode::Enter, m)
            if m.contains(KeyModifiers::SHIFT) || m.contains(KeyModifiers::ALT) =>
        {
            app.input.push('\n');
        }
        (KeyCode::Enter, _) => {
            let text = app.input.trim().to_string();
            if text.is_empty() {
                return;
            }
            let Some(sid) = app.active_session.clone() else { return };
            app.input.clear();
            app.drafts.remove(&sid);
            if let Some(rest) = text.strip_prefix('/') {
                // The UI parses `/name rest`; the handler decides what rest is.
                let (name, args) = rest.split_once(' ').unwrap_or((rest, ""));
                out.push(Command::CommandRun {
                    name: name.to_string(),
                    args: args.to_string(),
                    session_id: Some(sid),
                });
            } else {
                out.push(Command::TurnSubmit {
                    session_id: sid,
                    text,
                    forced_skill: None,
                });
            }
        }
        (KeyCode::Backspace, _) => {
            app.input.pop();
        }
        (KeyCode::Char(c), m) if !m.contains(KeyModifiers::CONTROL) => {
            app.input.push(c);
        }
        (KeyCode::Up, _) => {
            // Leaving the entry upwards puts the cursor on the last row, which
            // is where the conversation is.
            app.chat_focus = ChatFocus::Log;
            app.chat_cursor = None;
        }
        _ => {}
    }
}

fn decision_reason_key(app: &mut App, key: KeyEvent) -> Vec<Command> {
    let mut out = Vec::new();
    let Some(d) = &mut app.decision else { return out };
    let Some(reason) = &mut d.reason else { return out };
    match key.code {
        // Empty is allowed — that is the plain refusal.
        KeyCode::Enter => {
            let reason = reason.clone();
            let sid = d.session_id.clone();
            app.decision = None;
            app.touch();
            out.push(Command::DecisionResolve {
                session_id: sid,
                approved: false,
                reason,
            });
        }
        KeyCode::Esc => {
            d.reason = None;
        }
        KeyCode::Backspace => {
            reason.pop();
        }
        KeyCode::Char(c) => reason.push(c),
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{ChatLayout, Decision, Working};
    use crate::protocol::Entry;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn session_app() -> App {
        App {
            active_session: Some("s1".into()),
            column: Column::Chat,
            chat_focus: ChatFocus::Input,
            ..Default::default()
        }
    }

    #[test]
    fn enter_sends_and_clears_the_draft() {
        let mut app = session_app();
        app.input = "hello".into();
        let cmds = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Enter));
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].type_name(), "turn.submit");
        assert!(app.input.is_empty());
    }

    #[test]
    fn a_slash_line_becomes_a_command_not_a_message() {
        let mut app = session_app();
        app.input = "/compact fold it".into();
        let cmds = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Enter));
        assert_eq!(cmds[0].type_name(), "command.run");
        let p = cmds[0].payload();
        assert_eq!(p["name"], "compact");
        assert_eq!(p["args"], "fold it");
    }

    #[test]
    fn shift_enter_makes_a_new_line_instead_of_sending() {
        let mut app = session_app();
        app.input = "one".into();
        let cmds = handle_key(
            &mut app,
            &ChatLayout::default(),
            KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
        );
        assert!(cmds.is_empty());
        assert_eq!(app.input, "one\n");
    }

    #[test]
    fn one_escape_does_not_stop_a_turn_but_two_do() {
        let mut app = session_app();
        app.chat_focus = ChatFocus::Log;
        app.working = Some(Working {
            activity: "thinking".into(),
            started: Instant::now(),
            interruptible: true,
        });
        let first = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Esc));
        assert!(first.is_empty(), "a lone escape must not interrupt");
        let second = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Esc));
        assert_eq!(second[0].type_name(), "turn.interrupt");
    }

    #[test]
    fn escape_in_the_entry_leaves_the_entry_rather_than_stopping_the_agent() {
        let mut app = session_app();
        app.working = Some(Working {
            activity: "thinking".into(),
            started: Instant::now(),
            interruptible: true,
        });
        let cmds = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Esc));
        assert!(cmds.is_empty());
        assert_eq!(app.chat_focus, ChatFocus::Log);
    }

    #[test]
    fn q_reaches_the_entry_as_a_letter() {
        let mut app = session_app();
        handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Char('q')));
        assert!(!app.should_quit, "q must type, not quit, while the entry has focus");
        assert_eq!(app.input, "q");
    }

    #[test]
    fn q_quits_from_the_sessions_column_where_no_typing_happens() {
        let mut app = App { column: Column::Sessions, ..Default::default() };
        handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn the_approval_bar_answers_before_the_column_does() {
        let mut app = session_app();
        app.decision = Some(Decision {
            session_id: "s1".into(),
            question: "run this?".into(),
            detail: "rm -rf /".into(),
            reason: None,
        });
        let cmds = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Char('y')));
        assert_eq!(cmds[0].type_name(), "decision.resolve");
        assert_eq!(cmds[0].payload()["approved"], true);
        assert!(app.decision.is_none());
    }

    #[test]
    fn a_refusal_with_a_reason_sends_the_finished_string_only() {
        let mut app = session_app();
        app.decision = Some(Decision {
            session_id: "s1".into(),
            question: "run this?".into(),
            detail: String::new(),
            reason: None,
        });
        handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Char('r')));
        for c in "no".chars() {
            handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Char(c)));
        }
        let cmds = handle_key(&mut app, &ChatLayout::default(), key(KeyCode::Enter));
        let p = cmds[0].payload();
        assert_eq!(p["approved"], false);
        assert_eq!(p["reason"], "no");
    }

    #[test]
    fn shift_tab_cycles_the_mode_and_tells_the_core() {
        let mut app = session_app();
        let cmds = handle_key(
            &mut app,
            &ChatLayout::default(),
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
        );
        assert_eq!(app.mode, "auto");
        assert_eq!(cmds[0].payload()["mode"], "auto");
    }

    #[test]
    fn enter_on_a_thinking_row_toggles_it() {
        let mut app = session_app();
        app.chat_focus = ChatFocus::Log;
        app.entries = vec![Entry { kind: "thinking".into(), steps: 1, ..Default::default() }];
        let mut layout = ChatLayout::default();
        layout.rebuild(&app, 60);
        app.chat_cursor = Some(0);
        handle_key(&mut app, &layout, key(KeyCode::Enter));
        assert!(app.expansion.get(&0).unwrap().open);
    }
}
