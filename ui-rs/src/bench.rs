//! Seeing and timing the front-end without a terminal.
//!
//! `TestBackend` renders into an in-memory buffer, which is what makes both of
//! these possible in a headless job: `--snapshot` prints the frame as text, and
//! `--bench` times frames against a synthetic transcript.
//!
//! What `--bench` measures is the cost of *one drawn frame while a turn is in
//! flight* — the path `WorkingIndicator._render_frame` names as the problem:
//! at 12.5 frames a second Textual re-laid out the whole log, O(messages), and
//! measured loop lag went from 0.7ms to 44ms (p95) at 300 messages.
//!
//! The two numbers are not the same statistic — that one is event-loop lag,
//! this one is time in `draw` — so this does not claim "44ms became X". What it
//! does establish is the *shape*: whether frame cost grows with the length of
//! the conversation, which is the property that made the Textual version get
//! worse the longer you used it.

use std::time::{Duration, Instant};

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use crate::app::{App, ChatFocus, ChatLayout, Column, Expansion, Working};
use crate::protocol::{Entry, PanelRow, Part, SessionRow};

const WIDTH: u16 = 120;
const HEIGHT: u16 = 40;

/// A transcript with the shapes that actually occur: user turns, replies,
/// thinking boxes with tool exchanges folded in, and an error.
pub fn synthetic(messages: usize) -> App {
    let mut app = App {
        profile: "default".into(),
        version: "v0.25.0".into(),
        active_session: Some("s1".into()),
        mode: "auto".into(),
        model: "qwen3-coder-30b @ localhost:20001".into(),
        context_used: 24_600,
        context_window: 32_768,
        column: Column::Chat,
        chat_focus: ChatFocus::Input,
        ..Default::default()
    };
    app.sessions = vec![
        SessionRow {
            session_id: "s1".into(),
            title: "align the reads".into(),
            profile: "default".into(),
            mode: "auto".into(),
            flags: vec!["working".into()],
        },
        SessionRow {
            session_id: "s2".into(),
            title: "why did 4417 die".into(),
            profile: "default".into(),
            mode: "manual".into(),
            flags: vec![],
        },
    ];
    app.panel = vec![
        PanelRow {
            key: "w1".into(),
            title: "job 4417".into(),
            text: "RUNNING  12:04 elapsed\nnode-023  32 cpus".into(),
            kind: crate::protocol::PANEL_WATCH.into(),
            r#ref: "1".into(),
            ..Default::default()
        },
        PanelRow {
            key: "w2".into(),
            title: "align.err".into(),
            text: "quiet for 3m".into(),
            kind: crate::protocol::PANEL_WATCH.into(),
            r#ref: "2".into(),
            ..Default::default()
        },
    ];

    for i in 0..messages {
        match i % 4 {
            0 => app.entries.push(Entry {
                kind: "user".into(),
                text: format!("message {i}: can you check whether the run finished cleanly?"),
                index: i as i64,
                ..Default::default()
            }),
            1 => app.entries.push(Entry {
                kind: "thinking".into(),
                steps: 2,
                reasoning_chars: 1840,
                parts: vec![
                    Part {
                        kind: "reasoning".into(),
                        text: "The log tail is the fastest check, then squeue.".into(),
                        ..Default::default()
                    },
                    Part {
                        kind: "call".into(),
                        tool: "run_bash".into(),
                        target: "align.err".into(),
                        text: "tail -n 40 /scratch/run/align.err".into(),
                        result: "slurmstepd: error: JOB 4417 CANCELLED DUE TO TIME LIMIT"
                            .into(),
                        done: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            2 => app.entries.push(Entry {
                kind: "assistant".into(),
                text: format!(
                    "reply {i}: it hit the wall clock — the step was cancelled at the \
                     time limit rather than failing, so the outputs up to that point \
                     are intact and the run can be resumed from the last checkpoint."
                ),
                index: i as i64,
                ..Default::default()
            }),
            _ => app.entries.push(Entry {
                kind: "event".into(),
                text: format!("job 44{i:02} finished (exit 0)"),
                ..Default::default()
            }),
        }
    }
    // Every fifth thinking box left open, so the bench is not measuring the
    // cheapest possible log.
    for i in (1..messages).step_by(20) {
        app.expansion.insert(i, Expansion { open: true, parts: Default::default() });
    }
    app
}

/// Render one frame off-screen and print it, so a headless run can see the UI.
pub fn snapshot(messages: usize) {
    let mut app = synthetic(messages);
    app.working = Some(Working {
        activity: "run_bash".into(),
        started: Instant::now(),
        interruptible: true,
    });
    app.notify("warning", "context is past 70% — /compact folds it", Some(30.0));

    let mut term = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
    let mut layout = ChatLayout::default();
    term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 3)).unwrap();

    let buf = term.backend().buffer();
    println!("┌{}┐", "─".repeat(WIDTH as usize));
    for y in 0..HEIGHT {
        let mut line = String::new();
        for x in 0..WIDTH {
            line.push_str(buf[(x, y)].symbol());
        }
        println!("│{}│", line.trim_end().to_string() + &" ".repeat(
            (WIDTH as usize).saturating_sub(crate::app::display_width(line.trim_end()))
        ));
    }
    println!("└{}┘", "─".repeat(WIDTH as usize));
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Time frames at a range of transcript lengths and print the table.
pub fn bench(max_messages: usize) {
    println!("hpca-ui — frame cost while a turn is in flight");
    println!("terminal {WIDTH}x{HEIGHT}, spinner running, 500 frames per row\n");
    println!(
        "{:>9}  {:>7}  {:>9}  {:>9}  {:>9}  {:>11}",
        "messages", "lines", "layout", "frame p50", "frame p95", "rebuilds"
    );
    println!("{}", "─".repeat(66));

    let sizes: Vec<usize> = [10usize, 50, 100, 300, 1000, 3000]
        .into_iter()
        .filter(|n| *n <= max_messages.max(10))
        .collect();

    for n in sizes {
        let mut app = synthetic(n);
        app.working = Some(Working {
            activity: "run_bash".into(),
            started: Instant::now(),
            interruptible: true,
        });
        let mut term = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
        let mut layout = ChatLayout::default();

        // The cold cost: the O(messages) pass, done once.
        let t0 = Instant::now();
        layout.rebuild(&app, WIDTH - 2);
        let cold = t0.elapsed();
        let lines = layout.total_lines();

        // Warm the terminal's own diff so the first frame is not counted as a
        // full repaint of the screen.
        term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 0)).unwrap();

        let rebuilds_before = REBUILDS.with(|c| c.get());
        let mut samples = Vec::with_capacity(500);
        for tick in 0..500u64 {
            let t = Instant::now();
            term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, tick + 1))
                .unwrap();
            samples.push(t.elapsed());
        }
        let rebuilds = REBUILDS.with(|c| c.get()) - rebuilds_before;
        samples.sort();

        println!(
            "{:>9}  {:>7}  {:>7.2}ms  {:>7.3}ms  {:>7.3}ms  {:>11}",
            n,
            lines,
            cold.as_secs_f64() * 1000.0,
            percentile(&samples, 0.50).as_secs_f64() * 1000.0,
            percentile(&samples, 0.95).as_secs_f64() * 1000.0,
            rebuilds,
        );
    }

    println!(
        "\nlayout = one full rebuild (what a resize costs). frame = one drawn frame\n\
         with the spinner advancing, which is the path that made the Textual\n\
         version degrade with conversation length. rebuilds = layout passes\n\
         triggered across those 500 frames; 0 is the point."
    );
}

// A counter the layout bumps, so the bench can assert that a spinner frame
// really is not rebuilding anything rather than merely being fast.
thread_local! {
    pub static REBUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_renders_the_whole_screen_without_panicking() {
        let mut app = synthetic(40);
        let mut term = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
        let mut layout = ChatLayout::default();
        term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 1)).unwrap();
        let buf = term.backend().buffer();
        let text: String = (0..HEIGHT)
            .map(|y| {
                (0..WIDTH)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Sessions"));
        assert!(text.contains("Chat"));
        assert!(text.contains("Watchers"));
        assert!(text.contains("mode: auto"));
    }

    /// A resize to ~50 columns used to take the whole Textual app down: a
    /// bordered widget with zero content width crashed Rich's wrapping.
    #[test]
    fn a_very_narrow_terminal_does_not_crash_the_front_end() {
        let mut app = synthetic(20);
        for w in [20u16, 30, 40, 50, 74] {
            let mut term = Terminal::new(TestBackend::new(w, 20)).unwrap();
            let mut layout = ChatLayout::default();
            term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 1)).unwrap();
        }
    }

    #[test]
    fn a_very_short_terminal_does_not_crash_the_front_end() {
        let mut app = synthetic(20);
        for h in [3u16, 5, 8] {
            let mut term = Terminal::new(TestBackend::new(80, h)).unwrap();
            let mut layout = ChatLayout::default();
            term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 1)).unwrap();
        }
    }

    /// The claim this prototype rests on, asserted rather than asserted-at.
    #[test]
    fn frame_cost_does_not_grow_with_the_length_of_the_conversation() {
        let time_at = |n: usize| -> Duration {
            let mut app = synthetic(n);
            app.working = Some(Working {
                activity: "run_bash".into(),
                started: Instant::now(),
                interruptible: true,
            });
            let mut term = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
            let mut layout = ChatLayout::default();
            term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, 0)).unwrap();
            let mut total = Duration::ZERO;
            for tick in 0..200u64 {
                let t = Instant::now();
                term.draw(|f| crate::ui::draw(f, &mut app, &mut layout, tick + 1))
                    .unwrap();
                total += t.elapsed();
            }
            total / 200
        };
        let small = time_at(20);
        let large = time_at(2000);
        // 100x the messages. Generous bound: this is a timing test on a shared
        // box, and the point is the absence of O(n), not a precise ratio.
        assert!(
            large < small * 8 + Duration::from_millis(1),
            "frame cost grew with transcript length: {small:?} at 20 messages, \
             {large:?} at 2000"
        );
    }
}
