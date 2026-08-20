//! Drawing. The Textual layout, one for one: a top bar, three columns
//! (sessions | chat | watchers) in a 1:2:1 split, and a footer of key hints.
//!
//! Nothing here allocates per chat entry. The log is drawn by copying the
//! visible slice of [`ChatLayout::lines`], so the cost of a frame is the height
//! of the terminal and not the length of the conversation — which is the single
//! behaviour this prototype exists to demonstrate.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{App, ChatFocus, ChatLayout, ChatRow, Column, LineKind};

/// Mirrors `WorkingIndicator.FRAMES`.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// The palette stands in for Textual's `$accent` / `$success` / `$warning`.
const ACCENT: Color = Color::Cyan;
const SUCCESS: Color = Color::Green;
const WARNING: Color = Color::Yellow;
const ERROR: Color = Color::Red;
const MUTED: Color = Color::DarkGray;
const PANEL: Color = Color::Gray;

/// Mirrors `context_bar.BAR_CELLS` and its two thresholds.
const BAR_CELLS: usize = 28;
const WARN_FRACTION: f64 = 0.70;
const DANGER_FRACTION: f64 = 0.90;

pub fn draw(f: &mut Frame, app: &mut App, layout: &mut ChatLayout, tick: u64) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // top bar
            Constraint::Min(3),    // columns
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_top_bar(f, app, rows[0]);

    // Narrower than this the columns keep their sizes and the terminal clips
    // them, rather than the chat column being squeezed to nothing — the same
    // guard the Textual `#columns { min-width: 74 }` rule exists for.
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Fill(1),
            Constraint::Fill(2),
            Constraint::Fill(1),
        ])
        .split(rows[1]);

    draw_sessions(f, app, cols[0]);
    draw_chat(f, app, layout, cols[1], tick);
    draw_watchers(f, app, cols[2]);
    draw_footer(f, app, rows[2]);
    draw_toasts(f, app, area);
}

fn column_block(title: &str, focused: bool) -> Block<'_> {
    let border = if focused {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(PANEL)
    };
    Block::default()
        .title(Line::from(Span::styled(
            title,
            Style::default().add_modifier(Modifier::BOLD),
        )))
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_type(if focused {
            BorderType::Thick
        } else {
            BorderType::Plain
        })
        .border_style(border)
}

fn draw_top_bar(f: &mut Frame, app: &App, area: Rect) {
    let text = format!(
        " HPCA {} │ profile: {} │ (c) config",
        app.version, app.profile
    );
    f.render_widget(
        Paragraph::new(text).style(Style::default().bg(Color::Blue).fg(Color::White)),
        area,
    );
}

fn draw_sessions(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.column == Column::Sessions;
    let block = column_block("Sessions", focused);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::with_capacity(app.sessions.len());
    for (i, s) in app.sessions.iter().enumerate() {
        let selected = i == app.session_cursor;
        // The core owns these markers because it owns the state behind them.
        let glyph = if s.flags.iter().any(|fl| fl == "working") {
            SPINNER[0]
        } else if s.flags.iter().any(|fl| fl == "decision") {
            "?"
        } else {
            " "
        };
        let title = if s.title.is_empty() {
            "(untitled)"
        } else {
            &s.title
        };
        let style = if selected && focused {
            Style::default().bg(ACCENT).fg(Color::Black)
        } else if selected {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{glyph} {title}"), style),
            Span::styled(format!("  {}", s.mode), Style::default().fg(MUTED)),
        ]));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            " (no sessions — n for a new one)",
            Style::default().fg(MUTED),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_watchers(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.column == Column::Watchers;
    let block = column_block("Watchers", focused);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in app.panel.iter().enumerate() {
        let selected = i == app.panel_cursor;
        let style = if selected && focused {
            Style::default().bg(ACCENT).fg(Color::Black)
        } else if selected {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        if !row.title.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("─ {} ", row.title),
                Style::default().fg(MUTED),
            )));
        }
        for l in crate::app::wrap(&row.text, inner.width.max(4) as usize) {
            lines.push(Line::from(Span::styled(l, style)));
        }
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            " (nothing watched)",
            Style::default().fg(MUTED),
        )));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_chat(f: &mut Frame, app: &mut App, layout: &mut ChatLayout, area: Rect, tick: u64) {
    let focused = app.column == Column::Chat;
    let block = column_block("Chat", focused);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let has_session = app.active_session.is_some();
    // The entry only exists inside a session, so an empty chat column cannot
    // invite typing that has nowhere to go.
    let input_height = if has_session {
        (crate::app::wrap(&app.input, inner.width.max(4) as usize).len() as u16)
            .clamp(1, 10)
            + 2
    } else {
        0
    };
    let decision_height = if app.decision.is_some() { 6 } else { 0 };
    let chrome = if has_session { 2 } else { 0 }; // model line + context bar
    let mode_height = if has_session { 1 } else { 0 };

    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(chrome),
            Constraint::Min(1),
            Constraint::Length(decision_height),
            Constraint::Length(mode_height),
            Constraint::Length(input_height),
        ])
        .split(inner);

    if has_session {
        draw_chat_chrome(f, app, parts[0]);
    }
    draw_log(f, app, layout, parts[1], tick);
    if app.decision.is_some() {
        draw_decision(f, app, parts[2]);
    }
    if has_session {
        draw_mode(f, app, parts[3]);
        draw_input(f, app, parts[4]);
    }
}

fn draw_chat_chrome(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    let model = if app.model.is_empty() {
        "model: —".to_string()
    } else {
        format!("model: {}", app.model)
    };
    f.render_widget(
        Paragraph::new(model).style(Style::default().fg(MUTED)),
        rows[0],
    );
    f.render_widget(
        Paragraph::new(context_text(app)).style(Style::default().fg(context_colour(app))),
        rows[1],
    );
}

/// The port of `context_bar.render_bar`.
pub fn context_text(app: &App) -> String {
    // "~" marks a figure derived from character counts rather than measured by
    // the backend, so a number only roughly right never looks exact.
    let mark = if app.context_estimated { "~" } else { "" };
    let used = crate::protocol::thousands(app.context_used);
    if app.context_window <= 0 {
        return format!("context: {mark}{used} tokens used · window unknown");
    }
    let fraction = (app.context_used as f64 / app.context_window as f64).min(1.0);
    let filled = (fraction * BAR_CELLS as f64).round() as usize;
    let bar = "█".repeat(filled) + &"─".repeat(BAR_CELLS - filled);
    format!(
        "context [{bar}] {mark}{used} / {} ({:.0}%)",
        crate::protocol::thousands(app.context_window),
        fraction * 100.0
    )
}

fn context_colour(app: &App) -> Color {
    if app.context_window <= 0 {
        return MUTED;
    }
    let fraction = app.context_used as f64 / app.context_window as f64;
    if fraction >= DANGER_FRACTION {
        ERROR
    } else if fraction >= WARN_FRACTION {
        WARNING
    } else {
        MUTED
    }
}

fn draw_log(f: &mut Frame, app: &mut App, layout: &mut ChatLayout, area: Rect, tick: u64) {
    layout.ensure(app, area.width);
    let height = area.height as usize;
    let total = layout.total_lines();

    // Keep the cursor's row on screen, and otherwise stay pinned to the foot of
    // the log — a conversation the user has not scrolled follows the reply.
    let mut top = app.scroll as usize;
    if let Some(cur) = app.chat_cursor {
        if let Some((s, e)) = layout.row_span(cur) {
            if s < top {
                top = s;
            } else if e > top + height {
                top = e.saturating_sub(height);
            }
        }
    } else {
        top = total.saturating_sub(height);
    }
    top = top.min(total.saturating_sub(height.min(total)));
    app.scroll = top as u16;

    let cursor_row = if app.chat_focus == ChatFocus::Log && app.column == Column::Chat {
        app.chat_cursor
    } else {
        None
    };

    let mut out: Vec<Line> = Vec::with_capacity(height);
    for line in layout.lines.iter().skip(top).take(height) {
        // The spinner's text is written here, not in the layout: that is what
        // makes a frame cost one line instead of a pass over the log.
        let (text, kind) = if line.kind == LineKind::Working {
            (working_text(app, tick), LineKind::Working)
        } else {
            (line.text.clone(), line.kind)
        };
        let mut style = style_for(kind);
        if Some(line.row) == cursor_row {
            style = style.add_modifier(Modifier::REVERSED);
        }
        out.push(Line::from(Span::styled(text, style)));
    }

    if out.is_empty() {
        out.push(Line::from(Span::styled(
            if app.active_session.is_some() {
                " (no messages yet)"
            } else {
                " (open a session on the left)"
            },
            Style::default().fg(MUTED),
        )));
    }
    f.render_widget(Paragraph::new(out), area);
}

/// The port of `WorkingIndicator._frame_text`.
pub fn working_text(app: &App, tick: u64) -> String {
    let Some(w) = &app.working else {
        return String::new();
    };
    let frame = SPINNER[(tick as usize) % SPINNER.len()];
    let elapsed = w.elapsed_secs();
    let elapsed = if elapsed > 0 {
        format!(" {elapsed}s")
    } else {
        String::new()
    };
    // Both routes are named because they are not interchangeable: enter has to
    // be aimed at this line, esc esc works from wherever the user already is.
    let hint = if w.interruptible {
        "  (enter or esc esc to interrupt)"
    } else {
        ""
    };
    format!("{frame} {}…{elapsed}{hint}", w.activity)
}

fn style_for(kind: LineKind) -> Style {
    match kind {
        LineKind::UserBorder => Style::default().fg(ACCENT),
        LineKind::UserText => Style::default(),
        LineKind::AssistantBorder => Style::default().fg(SUCCESS),
        LineKind::AssistantText => Style::default(),
        LineKind::ErrorBorder | LineKind::ErrorText => Style::default().fg(ERROR),
        LineKind::ThinkingHeader => Style::default().fg(MUTED),
        LineKind::EventText => Style::default().fg(Color::Magenta),
        LineKind::RecallText => Style::default().fg(Color::Blue),
        LineKind::StepText => Style::default().fg(MUTED),
        LineKind::StepCall => Style::default().fg(WARNING),
        LineKind::Working => Style::default().fg(ACCENT),
        LineKind::Blank => Style::default(),
    }
}

/// A parked turn's approval renders at the foot of the chat log for the session
/// it belongs to — never as a modal over the whole TUI.
fn draw_decision(f: &mut Frame, app: &App, area: Rect) {
    let Some(d) = &app.decision else { return };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(WARNING))
        .title(" approval needed ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines = vec![Line::from(Span::styled(
        d.question.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    for l in crate::app::wrap(&d.detail, inner.width.max(4) as usize).into_iter().take(2) {
        lines.push(Line::from(Span::styled(l, Style::default().fg(MUTED))));
    }
    match &d.reason {
        Some(reason) => lines.push(Line::from(vec![
            Span::styled("why not: ", Style::default().fg(MUTED)),
            Span::raw(reason.clone()),
            Span::styled("▏", Style::default().fg(ACCENT)),
            Span::styled("  (enter sends, esc cancels)", Style::default().fg(MUTED)),
        ])),
        None => lines.push(Line::from(vec![
            Span::styled(" y ", Style::default().bg(SUCCESS).fg(Color::Black)),
            Span::raw(" approve   "),
            Span::styled(" n ", Style::default().bg(ERROR).fg(Color::Black)),
            Span::raw(" refuse   "),
            Span::styled(" r ", Style::default().bg(WARNING).fg(Color::Black)),
            Span::raw(" refuse with a reason"),
        ])),
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_mode(f: &mut Frame, app: &App, area: Rect) {
    let hint = match app.mode.as_str() {
        "manual" => "scripts run only with your approval",
        "auto" => "works until the task is done",
        "full-auto" => "asks for nothing, destructive ops included",
        _ => "",
    };
    let colour = match app.mode.as_str() {
        "manual" => WARNING,
        "auto" => SUCCESS,
        "full-auto" => ERROR,
        _ => MUTED,
    };
    // "full-auto" reads as "full auto".
    let label = app.mode.replace('-', " ");
    f.render_widget(
        Paragraph::new(format!(
            " mode: {label} — {hint} · shift+tab to switch"
        ))
        .style(Style::default().fg(colour)),
        area,
    );
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.column == Column::Chat && app.chat_focus == ChatFocus::Input;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if focused { ACCENT } else { PANEL }));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.input.is_empty() && !focused {
        f.render_widget(
            Paragraph::new("Message the agent…").style(Style::default().fg(MUTED)),
            inner,
        );
        return;
    }
    let wrapped = crate::app::wrap(&app.input, inner.width.max(4) as usize);
    let mut lines: Vec<Line> = wrapped.iter().map(|l| Line::from(l.clone())).collect();
    if focused {
        // A block cursor drawn into the text, so the prototype needs no
        // terminal cursor positioning to look alive.
        if let Some(last) = lines.last_mut() {
            last.spans.push(Span::styled("▏", Style::default().fg(ACCENT)));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    // (q) quit first, so the footer shows it leftmost.
    let mut keys: Vec<(&str, &str)> = Vec::new();
    if app.column == Column::Sessions {
        keys.push(("q", "quit"));
        keys.push(("enter", "open"));
        keys.push(("n", "new"));
        keys.push(("r", "rename"));
        keys.push(("t", "ask llm for a title"));
        keys.push(("d", "delete"));
    } else if app.column == Column::Watchers {
        keys.push(("d", "drop watch"));
    }
    keys.push(("c", "config"));
    keys.push(("shift+tab", "agent mode"));
    if app.working.is_some() {
        keys.push(("esc esc", "stop the agent"));
    }
    if app.chat_focus == ChatFocus::Input && app.column == Column::Chat {
        keys.push(("enter", "send"));
        keys.push(("esc", "the log"));
    }
    keys.push(("←/→", "column"));

    let mut spans: Vec<Span> = Vec::new();
    for (k, label) in keys {
        spans.push(Span::styled(
            format!(" {k} "),
            Style::default().bg(PANEL).fg(Color::Black),
        ));
        spans.push(Span::styled(
            format!(" {label}  "),
            Style::default().fg(MUTED),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_toasts(f: &mut Frame, app: &App, area: Rect) {
    if app.toasts.is_empty() {
        return;
    }
    let shown: Vec<_> = app.toasts.iter().rev().take(3).collect();
    let h = shown.len() as u16 + 2;
    let w = area.width.saturating_sub(4).min(60);
    let rect = Rect {
        x: area.width.saturating_sub(w + 2),
        y: 1,
        width: w,
        height: h.min(area.height),
    };
    f.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(MUTED));
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    let lines: Vec<Line> = shown
        .iter()
        .map(|t| {
            let colour = match t.severity.as_str() {
                "error" => ERROR,
                "warning" => WARNING,
                _ => SUCCESS,
            };
            Line::from(Span::styled(t.text.clone(), Style::default().fg(colour)))
        })
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
}

/// The row the cursor should land on when the log is entered from the entry
/// box: the last one, which is where the conversation is.
pub fn last_row(layout: &ChatLayout) -> Option<usize> {
    layout.rows.len().checked_sub(1)
}

/// Whether a row can be toggled open — thinking headers and their parts.
pub fn is_toggleable(app: &App, row: &ChatRow) -> bool {
    match row {
        ChatRow::Entry(i) => app
            .entries
            .get(*i)
            .map(|e| e.kind == "thinking")
            .unwrap_or(false),
        ChatRow::Part { .. } => true,
        ChatRow::Working => false,
    }
}
