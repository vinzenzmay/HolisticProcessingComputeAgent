//! What the front-end knows, and the row model the chat is drawn from.
//!
//! The whole point of this prototype lives in [`ChatLayout`]. Textual gives
//! every chat entry a widget, so anything that changes size — a spinner frame,
//! an expanded box — walks the widget tree and re-lays out the whole log,
//! O(messages). Here the log is flattened *once* into a vector of styled lines
//! and the renderer copies the visible slice out of it. A spinner frame touches
//! one line and costs nothing; a resize rebuilds the vector, which is the only
//! time the O(messages) work is actually needed.
//!
//! The rows are still selectable units, because the Textual original is a
//! `ListView` and up/down/enter move between *items*, not lines. So the layout
//! carries both: a row list for navigation and a line list for drawing, with an
//! index from one to the other.

use std::collections::HashMap;
use std::time::Instant;

use crate::protocol::{Entry, PanelRow, SessionRow};

/// Which column has the keyboard. Mirrors `action_focus_column`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Sessions,
    Chat,
    Watchers,
}

impl Column {
    pub fn shift(self, delta: i32) -> Column {
        let order = [Column::Sessions, Column::Chat, Column::Watchers];
        let at = order.iter().position(|c| *c == self).unwrap_or(1) as i32;
        let next = (at + delta).clamp(0, order.len() as i32 - 1);
        order[next as usize]
    }
}

/// Where a keypress goes inside the chat column: the log, or the entry box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFocus {
    Log,
    Input,
}

/// Whether a thinking box is open, and which of its parts are.
///
/// Held per entry index so an expanded view survives an unrelated repaint but
/// resets when a session is reloaded and fresh entries are built — the same
/// call `_ThinkingExpansion` makes.
#[derive(Debug, Clone, Default)]
pub struct Expansion {
    pub open: bool,
    pub parts: HashMap<usize, bool>,
}

/// A turn in flight on the session on screen.
#[derive(Debug, Clone)]
pub struct Working {
    pub activity: String,
    pub started: Instant,
    pub interruptible: bool,
}

impl Working {
    pub fn elapsed_secs(&self) -> u64 {
        self.started.elapsed().as_secs()
    }
}

/// A parked `interrupt()` waiting on the user.
#[derive(Debug, Clone)]
pub struct Decision {
    pub session_id: String,
    pub question: String,
    pub detail: String,
    /// The refusal reason being typed, once the user has chosen to explain one.
    pub reason: Option<String>,
}

/// A toast.
#[derive(Debug, Clone)]
pub struct Toast {
    pub severity: String,
    pub text: String,
    pub at: Instant,
    pub timeout: f64,
}

/// Everything the front-end draws.
pub struct App {
    pub profile: String,
    pub version: String,
    pub sessions: Vec<SessionRow>,
    pub session_cursor: usize,
    pub active_session: Option<String>,
    pub entries: Vec<Entry>,
    pub expansion: HashMap<usize, Expansion>,
    pub panel: Vec<PanelRow>,
    pub panel_cursor: usize,
    pub column: Column,
    pub chat_focus: ChatFocus,
    /// Which chat row the log cursor is on, if the log has ever been touched.
    pub chat_cursor: Option<usize>,
    pub scroll: u16,
    /// Unsent text per session — a draft must not follow the user into the
    /// next session, where it reads as that session's and is one Enter from
    /// being sent to the wrong thread.
    pub drafts: HashMap<String, String>,
    pub input: String,
    pub mode: String,
    pub model: String,
    pub context_used: i64,
    pub context_window: i64,
    pub context_estimated: bool,
    pub working: Option<Working>,
    pub decision: Option<Decision>,
    pub toasts: Vec<Toast>,
    pub status: String,
    pub connected: bool,
    /// Bumped whenever something that affects the chat lines changes, so the
    /// layout cache knows it is stale without diffing the transcript.
    pub revision: u64,
    pub should_quit: bool,
    /// When the last escape landed, for the two-press stop gesture.
    pub esc_armed_at: Option<Instant>,
    /// Frames drawn since start — the prototype's own render counter.
    pub frames: u64,
    /// Whether a `chat.append` landed during the turn now finishing. Guards
    /// against rendering the reply twice on a core that emits both it and
    /// `turn.finished.reply`.
    pub appended_this_turn: bool,
}

impl Default for App {
    fn default() -> Self {
        App {
            profile: String::new(),
            version: "proto".into(),
            sessions: Vec::new(),
            session_cursor: 0,
            active_session: None,
            entries: Vec::new(),
            expansion: HashMap::new(),
            panel: Vec::new(),
            panel_cursor: 0,
            column: Column::Chat,
            chat_focus: ChatFocus::Input,
            chat_cursor: None,
            scroll: 0,
            drafts: HashMap::new(),
            input: String::new(),
            mode: "manual".into(),
            model: String::new(),
            context_used: 0,
            context_window: 0,
            context_estimated: false,
            working: None,
            decision: None,
            toasts: Vec::new(),
            status: "connecting…".into(),
            connected: false,
            revision: 0,
            should_quit: false,
            esc_armed_at: None,
            frames: 0,
            appended_this_turn: false,
        }
    }
}

impl App {
    pub fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub fn selected_session(&self) -> Option<&SessionRow> {
        self.sessions.get(self.session_cursor)
    }

    pub fn selected_panel(&self) -> Option<&PanelRow> {
        self.panel.get(self.panel_cursor)
    }

    /// Park the on-screen draft under its session and load the next one's.
    pub fn swap_draft(&mut self, to: Option<&str>) {
        if let Some(cur) = self.active_session.clone() {
            if self.input.is_empty() {
                self.drafts.remove(&cur);
            } else {
                self.drafts.insert(cur, self.input.clone());
            }
        }
        self.input = to
            .and_then(|s| self.drafts.get(s).cloned())
            .unwrap_or_default();
    }

    pub fn notify(&mut self, severity: &str, text: &str, timeout: Option<f64>) {
        self.toasts.push(Toast {
            severity: severity.to_string(),
            text: text.to_string(),
            at: Instant::now(),
            timeout: timeout.unwrap_or(4.0),
        });
    }

    pub fn expire_toasts(&mut self) {
        self.toasts
            .retain(|t| t.at.elapsed().as_secs_f64() < t.timeout);
    }
}

// ------------------------------------------------------------- the row model

/// One selectable unit of the chat log — the port of a `ListView` child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRow {
    /// A whole entry: a message box, or a collapsed/expanded thinking header.
    Entry(usize),
    /// One revealed part of an expanded thinking box.
    Part { entry: usize, part: usize },
    /// The spinner at the foot of the log while a reply is on its way.
    Working,
}

/// How a line should be painted. Kept as an enum rather than a `Style` so the
/// layout stays independent of the terminal backend and can be asserted in a
/// test without a theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    UserBorder,
    UserText,
    AssistantBorder,
    AssistantText,
    ThinkingHeader,
    ErrorBorder,
    ErrorText,
    EventText,
    RecallText,
    StepText,
    StepCall,
    Working,
    Blank,
}

/// One drawn line of the chat log.
#[derive(Debug, Clone)]
pub struct ChatLine {
    pub text: String,
    pub kind: LineKind,
    /// Which row this line belongs to, so the cursor can highlight a whole row
    /// and scrolling can bring a row into view.
    pub row: usize,
}

/// The flattened log: rows for navigation, lines for drawing.
#[derive(Debug, Default)]
pub struct ChatLayout {
    pub rows: Vec<ChatRow>,
    pub lines: Vec<ChatLine>,
    /// First line index of each row, parallel to `rows`.
    pub row_starts: Vec<usize>,
    built_for_width: u16,
    built_for_revision: u64,
    built_with_working: bool,
}

impl ChatLayout {
    /// Rebuild only when the inputs that decide line breaks have changed.
    ///
    /// This is the cache that makes a spinner frame free: the spinner's own
    /// text is written at draw time, so a new frame does not invalidate
    /// anything here.
    pub fn ensure(&mut self, app: &App, width: u16) {
        let has_working = app.working.is_some();
        if self.built_for_width == width
            && self.built_for_revision == app.revision
            && self.built_with_working == has_working
            && !self.lines.is_empty()
        {
            return;
        }
        self.rebuild(app, width);
        self.built_for_width = width;
        self.built_for_revision = app.revision;
        self.built_with_working = has_working;
    }

    /// Force a rebuild — used by the benchmark, which wants the uncached cost.
    pub fn rebuild(&mut self, app: &App, width: u16) {
        // Counted so the bench can show that a spinner frame triggers none of
        // these, rather than merely being fast for some other reason.
        crate::bench::REBUILDS.with(|c| c.set(c.get() + 1));
        self.rows.clear();
        self.lines.clear();
        self.row_starts.clear();
        let inner = width.max(4) as usize;

        for (i, entry) in app.entries.iter().enumerate() {
            let row = self.rows.len();
            self.rows.push(ChatRow::Entry(i));
            self.row_starts.push(self.lines.len());
            match entry.kind.as_str() {
                "thinking" => self.push_thinking_header(app, entry, i, row, inner),
                "event" => self.push_plain(&entry.text, LineKind::EventText, row, inner),
                "recall" => self.push_plain(&entry.text, LineKind::RecallText, row, inner),
                kind => {
                    let (border, text, title) = match kind {
                        "user" => (LineKind::UserBorder, LineKind::UserText, "you"),
                        "error" => (LineKind::ErrorBorder, LineKind::ErrorText, "error"),
                        _ => (LineKind::AssistantBorder, LineKind::AssistantText, "agent"),
                    };
                    self.push_box(&entry.text, title, border, text, row, inner);
                }
            }

            // An expanded thinking box reveals each part as its own row below
            // the header, rather than spelling the whole box out inline.
            if entry.kind == "thinking" {
                if let Some(exp) = app.expansion.get(&i) {
                    if exp.open {
                        for (pi, part) in entry.parts.iter().enumerate() {
                            let prow = self.rows.len();
                            self.rows.push(ChatRow::Part { entry: i, part: pi });
                            self.row_starts.push(self.lines.len());
                            let open = exp.parts.get(&pi).copied().unwrap_or(false);
                            let kind = if part.kind == "call" {
                                LineKind::StepCall
                            } else {
                                LineKind::StepText
                            };
                            let marker = if open { "▼" } else { "▶" };
                            let rule = if part.kind == "call" { "▏" } else { " " };
                            self.push_line(
                                format!("  {rule} {marker} {}", part.label()),
                                kind,
                                prow,
                            );
                            if open {
                                self.push_line(String::new(), LineKind::Blank, prow);
                                for l in wrap(&part.body(), inner.saturating_sub(6)) {
                                    self.push_line(format!("  {rule}   {l}"), kind, prow);
                                }
                            }
                            self.push_line(String::new(), LineKind::Blank, prow);
                        }
                    }
                }
            }
        }

        if app.working.is_some() {
            let row = self.rows.len();
            self.rows.push(ChatRow::Working);
            self.row_starts.push(self.lines.len());
            // Placeholder: the renderer overwrites this line each frame, which
            // is precisely why a frame costs no layout.
            self.push_line(String::new(), LineKind::Working, row);
        }
    }

    fn push_line(&mut self, text: String, kind: LineKind, row: usize) {
        self.lines.push(ChatLine { text, kind, row });
    }

    fn push_plain(&mut self, text: &str, kind: LineKind, row: usize, inner: usize) {
        for l in wrap(text, inner) {
            self.push_line(l, kind, row);
        }
        self.push_line(String::new(), LineKind::Blank, row);
    }

    /// A titled round box — "who is speaking is a colour, not a prefix".
    fn push_box(
        &mut self,
        text: &str,
        title: &str,
        border: LineKind,
        body: LineKind,
        row: usize,
        inner: usize,
    ) {
        let w = inner.max(6);
        let content = w - 2;
        let title_bar: String = {
            let t = format!("╭─ {title} ");
            let pad = w.saturating_sub(display_width(&t) + 1);
            format!("{t}{}╮", "─".repeat(pad))
        };
        self.push_line(title_bar, border, row);
        for l in wrap(text, content.saturating_sub(2)) {
            let pad = content.saturating_sub(display_width(&l) + 2);
            self.push_line(format!("│ {l}{} │", " ".repeat(pad)), body, row);
        }
        self.push_line(format!("╰{}╯", "─".repeat(w - 2)), border, row);
        self.push_line(String::new(), LineKind::Blank, row);
    }

    fn push_thinking_header(
        &mut self,
        app: &App,
        entry: &Entry,
        idx: usize,
        row: usize,
        inner: usize,
    ) {
        let open = app.expansion.get(&idx).map(|e| e.open).unwrap_or(false);
        let marker = if open { "▼" } else { "▶" };
        let hint = if open {
            "enter to collapse"
        } else {
            "enter to expand"
        };
        let text = format!("{marker} {}  ({hint})", entry.summary());
        for l in wrap(&text, inner) {
            self.push_line(l, LineKind::ThinkingHeader, row);
        }
        if !open {
            self.push_line(String::new(), LineKind::Blank, row);
        }
    }

    pub fn total_lines(&self) -> usize {
        self.lines.len()
    }

    /// The line range a row occupies, for scrolling it into view.
    pub fn row_span(&self, row: usize) -> Option<(usize, usize)> {
        let start = *self.row_starts.get(row)?;
        let end = self
            .row_starts
            .get(row + 1)
            .copied()
            .unwrap_or(self.lines.len());
        Some((start, end))
    }
}

// ------------------------------------------------------------------ wrapping

pub fn display_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    s.width()
}

/// Break text to `width` columns, honouring the newlines already in it.
///
/// Word-wrapping, falling back to a hard break for a token longer than the
/// column — a 200-character path with no spaces in it is a real input here, and
/// dropping it off the edge would hide the thing the user is looking for.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for para in text.split('\n') {
        if para.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        for word in para.split(' ') {
            if word.is_empty() {
                continue;
            }
            let candidate = if line.is_empty() {
                display_width(word)
            } else {
                display_width(&line) + 1 + display_width(word)
            };
            if candidate <= width {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(word);
                continue;
            }
            if !line.is_empty() {
                out.push(std::mem::take(&mut line));
            }
            if display_width(word) <= width {
                line.push_str(word);
            } else {
                // Hard-break an over-long token by display columns.
                let mut chunk = String::new();
                for c in word.chars() {
                    let cw = display_width(&c.to_string());
                    if display_width(&chunk) + cw > width {
                        out.push(std::mem::take(&mut chunk));
                    }
                    chunk.push(c);
                }
                line = chunk;
            }
        }
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Part;

    fn app_with(entries: Vec<Entry>) -> App {
        App { entries, ..Default::default() }
    }

    fn msg(kind: &str, text: &str) -> Entry {
        Entry { kind: kind.into(), text: text.into(), ..Default::default() }
    }

    #[test]
    fn wrapping_never_drops_an_unbreakable_path() {
        let path = "/very/long/".to_string() + &"segment".repeat(40);
        let lines = wrap(&path, 20);
        let rejoined: String = lines.concat();
        assert_eq!(rejoined, path, "a hard break must lose nothing");
        assert!(lines.iter().all(|l| display_width(l) <= 20));
    }

    #[test]
    fn wrapping_keeps_the_newlines_it_was_given() {
        assert_eq!(wrap("a\n\nb", 10), vec!["a", "", "b"]);
    }

    #[test]
    fn every_line_belongs_to_the_row_it_came_from() {
        let app = app_with(vec![msg("user", "hello"), msg("assistant", "hi there")]);
        let mut layout = ChatLayout::default();
        layout.rebuild(&app, 40);
        assert_eq!(layout.rows.len(), 2);
        for line in &layout.lines {
            assert!(line.row < layout.rows.len());
        }
        // Each row's span must cover its own lines and no other's.
        for (r, _) in layout.rows.iter().enumerate() {
            let (s, e) = layout.row_span(r).unwrap();
            assert!(s <= e);
            for l in &layout.lines[s..e] {
                assert_eq!(l.row, r);
            }
        }
    }

    #[test]
    fn an_expanded_thinking_box_reveals_its_parts_as_rows() {
        let entry = Entry {
            kind: "thinking".into(),
            steps: 2,
            parts: vec![
                Part { kind: "reasoning".into(), text: "mulling".into(), ..Default::default() },
                Part {
                    kind: "call".into(),
                    tool: "run_bash".into(),
                    text: "ls".into(),
                    result: "a b".into(),
                    done: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut app = app_with(vec![entry]);
        let mut layout = ChatLayout::default();

        layout.rebuild(&app, 60);
        assert_eq!(layout.rows.len(), 1, "collapsed: only the header is a row");

        app.expansion.insert(0, Expansion { open: true, parts: HashMap::new() });
        layout.rebuild(&app, 60);
        assert_eq!(layout.rows.len(), 3, "expanded: the header plus its two parts");
        assert_eq!(layout.rows[1], ChatRow::Part { entry: 0, part: 0 });
        assert_eq!(layout.rows[2], ChatRow::Part { entry: 0, part: 1 });
    }

    #[test]
    fn a_spinner_frame_does_not_invalidate_the_layout() {
        // The whole reason for the cache: in Textual this path was a full
        // layout pass at 12.5fps, O(messages).
        let mut app = app_with((0..200).map(|i| msg("user", &format!("m{i}"))).collect());
        app.working = Some(Working {
            activity: "thinking".into(),
            started: Instant::now(),
            interruptible: true,
        });
        let mut layout = ChatLayout::default();
        layout.ensure(&app, 80);
        let built = layout.total_lines();
        // A frame advances only the spinner; nothing here changes.
        layout.ensure(&app, 80);
        assert_eq!(layout.total_lines(), built);
        assert_eq!(layout.built_for_revision, app.revision);
    }

    #[test]
    fn a_resize_is_what_earns_a_rebuild() {
        let app = app_with(vec![msg("assistant", &"word ".repeat(80))]);
        let mut layout = ChatLayout::default();
        layout.ensure(&app, 80);
        let wide = layout.total_lines();
        layout.ensure(&app, 30);
        assert!(layout.total_lines() > wide, "narrower must wrap to more lines");
    }

    #[test]
    fn a_draft_stays_with_the_session_it_was_typed_in() {
        let mut app = App { active_session: Some("a".into()), ..Default::default() };
        app.input = "half a thought".into();
        app.swap_draft(Some("b"));
        assert_eq!(app.input, "", "b has no draft of its own");
        app.active_session = Some("b".into());
        app.swap_draft(Some("a"));
        assert_eq!(app.input, "half a thought", "a's draft came back");
    }

    #[test]
    fn columns_stop_at_the_ends() {
        assert_eq!(Column::Sessions.shift(-1), Column::Sessions);
        assert_eq!(Column::Sessions.shift(1), Column::Chat);
        assert_eq!(Column::Watchers.shift(1), Column::Watchers);
    }
}
