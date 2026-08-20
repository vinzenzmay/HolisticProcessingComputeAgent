//! How frames get across — the twin of `hpca/transport.py`, client half only.
//!
//! Two failures are ordinary here rather than exceptional, because on this wire
//! they are: a frame that does not decode costs that frame and not the
//! connection, and a write to a core that has already gone is dropped instead
//! of panicking. The peer of a UI process is a thing that can die at any moment.
//!
//! Sequence numbers belong to the connection, not the caller. A caller who has
//! to remember to increment one is a caller who will eventually forget, and
//! nothing downstream can tell a skipped seq from a dropped frame.

use std::path::Path;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;

use crate::protocol::{Command, Envelope, Event};

/// Mirrors `transport.FRAME_LIMIT`. A `chat.reset` carries a whole transcript
/// and a tool result can be a file dump, so the ceiling is set where hitting it
/// means something is wrong rather than merely big.
pub const FRAME_LIMIT: usize = 16 * 1024 * 1024;

/// What the UI loop receives from the wire.
#[derive(Debug)]
pub enum Incoming {
    /// A frame that decoded and has a meaning.
    Event(Event),
    /// A frame that did not decode. Reported rather than swallowed so a
    /// mismatch between the two protocol definitions is visible in the UI
    /// instead of looking like a core that went quiet.
    Bad(String),
    /// The core closed the socket, or we did.
    Closed,
}

/// A live connection to a core: events out, commands in.
pub struct Connection {
    pub events: mpsc::UnboundedReceiver<Incoming>,
    commands: mpsc::UnboundedSender<Command>,
}

impl Connection {
    /// Queue a command. A send to a departed core is dropped, not raised —
    /// there is nothing a keypress handler could usefully do about it.
    pub fn send(&self, cmd: Command) {
        let _ = self.commands.send(cmd);
    }
}

/// Dial a core listening on `path` and start pumping both directions.
pub async fn connect_unix(path: &Path) -> Result<Connection> {
    let stream = UnixStream::connect(path)
        .await
        .with_context(|| format!("cannot reach a core at {}", path.display()))?;
    Ok(spawn_pumps(stream))
}

fn spawn_pumps(stream: UnixStream) -> Connection {
    let (read_half, mut write_half) = stream.into_split();
    let (ev_tx, ev_rx) = mpsc::unbounded_channel();
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Command>();

    // Reader: NDJSON in, typed events out.
    tokio::spawn(async move {
        let mut lines = BufReader::with_capacity(64 * 1024, read_half).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if line.len() > FRAME_LIMIT {
                        let _ = ev_tx.send(Incoming::Bad(format!(
                            "frame of {} bytes exceeds the {FRAME_LIMIT} limit",
                            line.len()
                        )));
                        continue;
                    }
                    let msg = match serde_json::from_str::<Envelope>(&line) {
                        Ok(env) => match Event::from_envelope(&env) {
                            Ok(ev) => Incoming::Event(ev),
                            Err(e) => Incoming::Bad(e.to_string()),
                        },
                        Err(e) => Incoming::Bad(format!("undecodable frame: {e}")),
                    };
                    if ev_tx.send(msg).is_err() {
                        break; // the UI is gone
                    }
                }
                Ok(None) => {
                    let _ = ev_tx.send(Incoming::Closed);
                    break;
                }
                Err(e) => {
                    let _ = ev_tx.send(Incoming::Bad(format!("read failed: {e}")));
                    let _ = ev_tx.send(Incoming::Closed);
                    break;
                }
            }
        }
    });

    // Writer: the connection owns the sequence counter.
    tokio::spawn(async move {
        let mut seq: i64 = 0;
        while let Some(cmd) = cmd_rx.recv().await {
            seq += 1;
            let env = cmd.to_envelope(seq, None);
            let Ok(mut line) = serde_json::to_vec(&env) else {
                continue; // only reachable for a payload holding a live object
            };
            line.push(b'\n');
            if write_half.write_all(&line).await.is_err() {
                break; // the core left; the reader will report Closed
            }
            let _ = write_half.flush().await;
        }
    });

    Connection { events: ev_rx, commands: cmd_tx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Command;
    use tokio::io::AsyncReadExt;

    /// A pasted traceback in a chat message is exactly the input that breaks a
    /// hand-rolled framing; JSON escapes the newline, so the delimiter cannot
    /// appear inside a frame.
    #[tokio::test]
    async fn a_newline_in_a_message_does_not_split_the_frame() {
        let (a, b) = UnixStream::pair().unwrap();
        let conn = spawn_pumps(a);
        conn.send(Command::TurnSubmit {
            session_id: "s1".into(),
            text: "line one\nline two".into(),
            forced_skill: None,
        });

        let mut buf = Vec::new();
        let mut reader = BufReader::new(b);
        // One frame ends at the first newline; if escaping were wrong we would
        // read a truncated, undecodable prefix here.
        let n = reader.read_until(b'\n', &mut buf).await.unwrap();
        assert!(n > 0);
        let env: Envelope = serde_json::from_slice(&buf[..n - 1]).unwrap();
        assert_eq!(env.kind, "turn.submit");
        assert_eq!(env.payload["text"], "line one\nline two");
        assert_eq!(env.seq, 1, "the connection numbers frames, not the caller");
    }

    #[tokio::test]
    async fn a_junk_frame_costs_that_frame_and_not_the_connection() {
        let (a, b) = UnixStream::pair().unwrap();
        let mut conn = spawn_pumps(a);
        let mut b = b;
        b.write_all(b"{not json}\n").await.unwrap();
        b.write_all(b"{\"type\":\"turn.started\",\"payload\":{\"session_id\":\"s1\"}}\n")
            .await
            .unwrap();

        match conn.events.recv().await.unwrap() {
            Incoming::Bad(_) => {}
            other => panic!("expected Bad, got {other:?}"),
        }
        match conn.events.recv().await.unwrap() {
            Incoming::Event(Event::TurnStarted { session_id }) => assert_eq!(session_id, "s1"),
            other => panic!("expected the next frame to survive, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_closed_peer_ends_the_stream() {
        let (a, b) = UnixStream::pair().unwrap();
        let mut conn = spawn_pumps(a);
        drop(b);
        match conn.events.recv().await.unwrap() {
            Incoming::Closed => {}
            other => panic!("expected Closed, got {other:?}"),
        }
    }
}
