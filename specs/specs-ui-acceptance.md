# The UI port's acceptance checklist

Companion to [specs-ui-replacement.md](specs-ui-replacement.md) §7.

Every claim below is a behaviour asserted today by one of the 34 Textual-coupled
test files. Each must end up asserted by a test against the new UI, or be
explicitly dropped with a reason written next to it. This list — not the old
test files — is what the port is measured against, because the old files assert
these behaviours *through* a framework that is going away.

The headings name the file the claims came from, so a rewrite can be checked
off file by file.

---

## Queueing while a turn runs — `test_tui_queue.py`

- A message sent while that session's turn is running is accepted, not refused.
- The input is cleared immediately so the user can keep typing.
- The queued message is visible in the transcript right away, as a `queued` entry.
- Only one turn per session runs at a time; extra messages queue for *that*
  session rather than starting a second turn on the same thread.
- The queue drains in order after the turn ends.
- A queued message survives the transcript rebuild triggered by the finishing
  turn's reply — it is not yet in the graph and must not vanish.
- A queued message survives the turn it waited for even when the queue drains
  *before* that turn's reply is drawn, i.e. the reply rebuilds from a snapshot
  older than the message.
- Slash commands are refused rather than queued: they act on the UI and run
  their own exclusive workers.
- A queued message can be cancelled — activating its row opens a dialog;
  cancelling drops it from both the queue and the log, with the text handed
  back to the entry to edit and re-send.
- Cancelling redraws the log without losing the working indicator for the turn
  still in flight.
- Escape on the cancel dialog leaves the message queued.
- Cancelling picks exactly the row that was selected: a message queued twice
  loses one copy, matched by position and not by text.
- A message that already started running while the dialog sat open is not
  cancelled — that is the working indicator's job.
- Switching session while the cancel dialog is open cancels nothing.
- A reply landing after the screen is gone (app tearing down) must not raise:
  keep the entries, stay quiet.
- A session parked on an approval does not stall the queue for a *different*
  session.

## Stopping a turn — `test_tui_interrupt.py`

- A working indicator is present while waiting on the model.
- The hint text follows what can actually be interrupted.
- A turn currently running a tool is interruptible.
- No interrupt dialog appears when not waiting on the LLM.
- Selecting the working indicator (arrow down onto the last log line, then
  enter) offers the confirm dialog.
- Confirming interrupts the turn and hands the user's message back into the entry.
- The handed-back message returns to *its own* session, not whichever session
  is now on screen; it waits there as a draft.
- Declining the dialog leaves the turn running.
- A reply landing while the dialog is open is a no-op.
- Typing ahead never buries the spinner: it stays the last, reachable line of
  the log while a turn runs.
- After an approval, the resumed turn carries the same interrupt anchor, so the
  spinner still stops it.
- The anchor is dropped once the exchange ends.
- A spinner that is not a turn — a silent `/conclude` or compaction call — says
  why it cannot be stopped rather than silently doing nothing.
- Escape-escape cancels the running turn from anywhere on the main screen and
  hands the message back.
- One escape changes nothing; terminals emit stray ESC.
- Two escapes far apart are two first presses: a timeout separates them.
- Escape-escape works from the sessions column too — no aiming required.
- Escape-escape stops a turn that is running a tool.
- Escape-escape breaks the decision-retry loop, where middleware rejects the
  model's output repeatedly.
- Escape on an open modal dismisses the modal and leaves the turn alone.
- Cancelling a turn does not kill a background job the turn started.

## Per-session concurrency — `test_tui_concurrent_turns.py`

- A turn held open in session A does not block a turn starting in B.
- Each turn resolves its own backend client; two sessions on different backends
  never see each other's user text.
- A second message in a session whose turn is running is queued for that
  session, not run as a second turn.
- Any session with a live turn shows an in-flight marker on its sidebar row.
- The marker clears when the turn completes.
- The marker shows for an off-screen session: start A, switch to B, A's row is
  still marked.
- The context meter always reflects the session on screen.
- A background turn silently updates its own stored token count; the visible
  meter never shows another session's number.
- Switching back to a session that ran off-screen shows *its* measured count.
- The on-screen session's own turn ticks the meter live.

## Replies to a session you left — `test_tui_background_reply.py`

- A reply landing in a session the user has left does not force that session
  open; it flags the sidebar row.
- Opening the flagged session shows the reply and clears the flag.
- The backgrounded turn logs into its own session transcript.
- The backgrounded session still gets its model-written title.
- A tool running in a backgrounded turn keeps its own session's registry,
  runner and session id — no cross-contamination from the switch.
- Sending to a different session while one is busy runs concurrently; typing is
  never blocked.
- The first turn survives the second being started.
- The spinner reappears when returning to the still-busy session.

## Rewind and fork — `test_tui_rewind.py`, `test_tui_message_reuse.py`

- Enter on one of your own sent messages opens the rewind dialog.
- Escape from the dialog changes nothing.
- Activating a sent message twice forks the session — the reflex key is
  the choice that cannot lose anything. On a queued message it still
  copies the text into the entry, which is all that dialog has to offer.
- `c` on the chat row copies the row under the cursor to the clipboard;
  the rewind dialog no longer offers a copy of its own.
- Rollback trims the conversation to before that message.
- Rollback to the first message empties the thread.
- The next turn runs on the trimmed thread: the model does not see the cut messages.
- Rollback is refused while a turn is running.
- Fork opens a *copy* of the session trimmed to before the message.
- The source session is untouched by a fork.
- Both sessions appear in the sidebar after a fork.
- A reused message appends on its own line to whatever is already typed; a
  draft ending in a space continues on the same line.
- The agent's reply is not copyable this way.
- A queued message is yours too; a background event is not.
- Reusing twice stacks both messages.
- A reused message becomes a draft belonging to the session it was taken from.

## Core chat wiring — `test_tui_chat.py`

- User and assistant messages appear in the log.
- The model names the session after the first exchange.
- When the model cannot title, the opening message is used.
- Later messages do not retitle.
- A second turn works in the same session.
- An LLM failure is shown as an error entry, not a crash.
- A destructive tool prompts inline; approving runs it, rejecting skips it.
- The model's own tool-call message is not drawn as a reply.
- Sessions are listed in the left column; selecting one loads its history and
  focuses its chat entry; the open session is highlighted.
- Creating a session via UI selection works.
- Up leaves the entry into the log and down returns to it; up in an empty log
  stays in the entry.
- Profile memories are injected into the system prompt.
- Memories are frozen per session and refresh only at boundaries, so the prompt
  prefix stays byte-stable for the backend's prefix cache; a note written
  elsewhere mid-session does not shift the prefix.
- User and assistant turns land in the episodic index and are recallable from a
  later session via `session_search` with no model call.
- Tool traffic is not indexed.
- The context bar reports the backend's measured `prompt_tokens`, not an estimate.
- The context bar shows turn speed: completion tokens over request wall clock.
- The context window is discovered from the backend, with no configuration.
- A small window turns the bar red.
- Reopening a session estimates usage before the next reply.
- Switching session does not show the previous session's context number.

## Inline approval prompts — `test_tui_approvals.py`

- The approval prompt is inline and focused, not a modal.
- Answering resumes the turn and clears the prompt.
- A decision pending in a background session flags the sidebar row without any
  overlay on the current session.
- A new session does not inherit another session's pending prompt.
- Opening the flagged session reveals the prompt.
- The prompt shows what *this call does*: the real path, the real diff, the
  real command.
- The tool's schema blurb and a raw JSON dump stay off the screen.
- With no `details` from the tool, arguments are shown as lines, not JSON.
- Plumbing arguments are omitted.
- The lines of a script shown below are not repeated above it.
- A pathological, huge argument is clipped.
- `details` replace the arguments entirely when present.

## Declining with a reason — `test_tui_decline_reason.py`

- `n` on a gated call opens a box for the reason and answers nothing yet.
- The call stays on screen while the reason is typed.
- The box takes the keys the prompt was using.
- Enter sends the typed reason back to the model with the refusal.
- An empty box is the plain refusal.
- Escape refuses without asking why.
- Escaping out of the reason box still refuses.
- Approving never asks why.
- A manually skipped script carries the reason back.
- A half-written reason waits in its own session across a session switch.

## Agent modes — `test_tui_modes.py`

- The mode bar is hidden without a session, shown with one.
- Shift+tab cycles the mode and the choice persists.
- New sessions start in the configured default mode.
- Manual mode gates `run_bash`; skipping reports back to the model.
- Approving in manual mode actually runs the script.
- Auto mode does not gate normal tools.
- Full-auto runs even destructive tools without approval.
- Shift+tab only cycles from the chat column; the sessions and watchers columns
  leave the mode alone.

## Reasoning box, tool rows, logging — `test_tui_thinking_logs.py`

- Reasoning and tool steps share one collapsed box per turn.
- Enter expands it into collapsible steps.
- Parts are individually navigable and hold the highlight.
- No reasoning means no box.
- The box survives reopening the session.
- An approved script stays readable in the chat after the prompt is gone.
- A skipped script is still there to read.
- Auto mode shows what ran unasked.
- The script survives reopening the session.
- The call is written to the session log file.
- The call is on screen *while* the tool runs — live steps, not only after.
- A live row opens without a box above it.
- A call alone is one row with no result on it.
- The result fills that same row rather than adding a second one.
- A result with no call of its own still gets a row.
- A rebuild between the two halves leaves the result standing.
- A turn is logged with reasoning and answer.
- Sub-agent query and reply are logged under the tool.
- Reopening a session does not re-log it.
- Each session gets its own log file.
- Errors are logged.
- Logging off writes nothing; switching logging off takes effect at once.

## The spinner — `test_tui_working.py`

- It appears while waiting and goes when the reply lands.
- It sits after the last message.
- It goes when the turn fails.
- At most one spinner per turn.
- It says the LLM is processing.
- It names the tool in flight.
- Steps are reported in order.
- Frames advance and the step is timed.
- A new step does not restart the clock; repeating the same step does not
  restart it either.
- Switching away and back keeps the elapsed time — "how long since I sent it".
- A backend call without a turn, e.g. `/conclude`, still times itself.
- A spinner frame must not relayout the whole log. (Carried over as a
  performance requirement from a bucket-C Textual test.)

## Titles, rename, delete — `test_tui_session_titles.py`

- The column shows the model's summary, not the raw first message.
- A session is titled once.
- Each new session is titled.
- A reopened session is not retitled.
- A slash command alone does not trigger a title.
- `r` opens a rename dialog prefilled with the current name and saves.
- Escape keeps the old name; an empty name is refused.
- A hand-written name is never overwritten by the model.
- `t` asks the model for a new title.
- Rename keys are inert on the `(new session)` row.
- A failed title call leaves the name alone.
- Retitling the open session parks the chat-bottom spinner labelled
  "writing a title".
- Retitling a highlighted-but-not-open session lights its sidebar row, never
  the chat spinner.
- Titling and renaming are logged.
- `d` asks, then deletes, and keeps the log file.
- Declining keeps the session.
- Deleting the open session empties the chat.
- Deleting another session leaves the open one alone.
- Chat history is dropped with the session.
- Delete is inert on the new-session row.

## Profiles — `test_tui_profiles.py`, `test_tui_editor_expansion.py`

- `a` opens the profiles screen only from the sessions column.
- The list shows profiles plus a `(new profile)` row, default marked.
- Escape returns to the main screen.
- Enter opens a profile's memories in the editor and keeps edits.
- Declining "keep?" discards edits; no edits means no question.
- Editing the active profile reaches the next turn's prompt.
- Enter on `(new profile)` prompts and creates; a bad name is reported and
  creates nothing.
- Deleting a profile reassigns its sessions to default.
- The default profile cannot be deleted; delete is inert on the new row.
- Deletion is blocked while a reply is in progress or a sub-process is running;
  an idle profile is deletable.
- A new session runs under the profile chosen in the picker.
- A row that names something and then says what it has — a profile and its
  counts, a skill and its description, a backend and its details — keeps at
  least two spaces between the two, whatever the name's length, and the column
  is counted in terminal cells rather than characters (`ansi.column`).
- The chosen profile's memories reach the prompt.
- Sessions of all profiles stay listed, tagged with their profile.
- Opening a session switches to its profile.
- Creating a profile from inside the picker uses it.
- A reused untouched session is retagged.
- Copying a profile inherits its memories and its skills; provenance is shown
  in the list; the two then diverge.
- The default is copyable though not deletable; copy is inert on the new row;
  a duplicate name is refused.
- Deleting a profile removes its skills.
- `r` on a profile row opens the RAG archive in the editor and keeps edits;
  inert on the `(new profile)` row.
- `s` lists and edits a profile's skills; `d` deletes one.

## Backends — `test_tui_llm_mgmt.py`, `test_tui_session_llm.py`, `test_tui_model_line.py`

- `q` from the sessions column asks to quit; `n` stays, `y` quits.
- ctrl+q does nothing — it belongs to zellij; `q` typed in chat is a letter;
  `q` is inert on the watchers column and on another screen.
- `m` opens manage-LLMs, only from the sessions column, and a scan populates
  the discovered panel.
- A cluster endpoint from the manifest reaches the discovered panel even though
  the localhost scan cannot see it.
- An empty localhost scan *with* a cluster hit is not treated as the off-cluster
  case: no tunnel lecture.
- Enter on a locked cluster row opens the key form with the model name prefilled.
- An empty scan with nothing configured shows the tunnel recipe in a selectable
  window that stays until escape.
- An empty scan with a backend up says only "nothing new".
- An empty scan with every backend down still gives the tunnel help.
- The help window waits for a form the user already opened rather than landing
  on top of it.
- Enter on a configured entry configures and persists it; configured entries
  show connection state.
- Enter on the configured panel does not set a global default.
- `r` removes a configured backend; denying keeps it; remove is inert on the
  discovered panel.
- Arrow keys switch panels; the footer offers "add" only on a discovered entry;
  escape closes.
- Discovered ports are remembered and persisted, and scanned first next time;
  a port is remembered when a backend is configured.
- ctrl+L without backends warns; it is only available in the chat column.
- Switching sets the *session's* backend and updates the model line; escape
  changes nothing.
- The UI stays responsive and closable during a slow scan.
- The discovered panel fills incrementally while the scan is still running.
- There is no per-backend reasoning marker or toggle any more; an old catalog
  entry carrying one still loads.
- The backend form: enter on a keyed endpoint opens with URL locked; key plus
  autofill saves; a rejected key warns and keeps the input; `a` opens a blank
  manual form that saves.
- Key registry: a re-probe upgrades a sentinel with a pooled key; an added keyed
  endpoint leaves no duplicate; a save-time guard drops a sentinel without a
  working key; a pool-unlocked entry is added without a form; a manual key feeds
  the pool and re-probes.
- Startup ends by probing the active backend: nothing answering opens
  manage-LLMs and says why; a key-locked backend counts as not connected; a
  working backend leaves the user alone; an auto-connected cluster LLM counts as
  connected, because the check runs after auto-connect.
- With no backends configured a session is created without a picker.
- Creation picks and stores the backend; cancelling the pick creates no session.
- `client_for` uses the session's backend, falling back to the bootstrap client.
- The context window follows the session's backend.
- The app's own sub-agent calls — `/conclude`, `/memorize`, titling — run on the
  session's own backend, defaulting to the active session, falling back to
  bootstrap.
- `ctx.llm`, a tool's firewalled sub-loop such as `ask_docs` or
  `explain_job_failure`, routes to the session's backend, including through the
  transcript-logging wrapper.
- The model line shows the active session's model, updates on session switch,
  and is hidden with no session open. The top bar keeps the profile and drops
  the model.

## Memory — `test_tui_memory.py`, `test_tui_memory_tool.py`

- `/memorize <note>` forms memories from the note plus the conversation; each
  still needs approval.
- The note and the conversation both reach the model.
- Rejecting saves nothing.
- `/memorize` with no note explains itself.
- `\memorize` (backslash) also works.
- A slash command is not used as the session's topic or title.
- An unknown command is reported, not sent to the model.
- The chat-bottom spinner shows while memories are being formed, labelled
  "forming memories", and clears on return.
- Typing `/` lists the commands; the list narrows as the name is typed; the
  menu goes when the draft becomes ordinary text; `\` lists them too.
- `/conclude` runs a full self-review — memories, struggle notes, skills — each
  approved via the reflection dialog; approve and reject are both honoured.
- The spinner shows during `/conclude`, labelled for the step.
- `/conclude` without a conversation warns.
- Over-cap memory is reported; under cap is quiet.
- Editor resolution order: settings, then `$VISUAL`, then `$EDITOR`, then nano.
- Editing a profile reloads it.
- A full system-prompt scope blocks new writes; room available allows them; RAG
  writes are never blocked.
- The curator runs at startup at most once every few days, archives old RAG
  entries, does not re-run on the second start, and can be disabled.
- Flagging with the `memory` tool writes nothing until `/conclude`.
- `/conclude` reviews and saves the flagged batch; rejecting saves nothing.
- A full scope rejects the flagged batch.
- A batch can free room and add in one call.
- A hand edit made during `/conclude` is not clobbered.
- Memory guidance is present in the prompt.
- The tool schema the app sends is self-contained, with no dangling `$defs`
  pointers — the failure that once broke every turn at the backend.
- A turn runs with the memory tool registered.

## Skills and self-review — `test_tui_skills_struggle.py`, `test_tui_skill_commands.py`

- Skills are not listed in the system prompt but `read_skill` is registered;
  `read_skill` returns the body.
- Shipped skills alone keep the tool registered and the "skills exist" note in
  the prompt.
- Reflection fires only on `/conclude`; there is no automatic review.
- `/conclude` saves a struggle note on approval; a rejected proposal is not saved.
- A clean turn triggers no review at all.
- `/conclude` on a clean session still proposes and saves — a stated preference
  is worth keeping.
- Nothing to save is silent.
- A skill patch appends to the profile's own copy.
- A new skill is created when enabled, and can be disabled.
- A matching struggle note is fenced into the model's copy of the user message;
  the stored transcript keeps the clean text.
- A non-matching request carries no memory block.
- RAG memories are not in the system prompt; a matching request retrieves them,
  an unrelated one retrieves nothing.
- Recall is visible in the transcript.
- Prefetch respects the char budget.
- Struggle notes from review land in RAG.
- `/skill-creator` creates a skill tied to the profile.
- `read_skill` is registered from the start, since HPCA ships skills, and a new
  skill does not disturb it.
- The skill list stays out of the system prompt.
- An empty name is rejected; a duplicate name is refused.
- `/skill-creator <what it should do>` drafts via the model into the same form;
  the user still edits and confirms.
- The pre-filled draft saves like any other skill.
- The request and the conversation reach the drafter.
- The spinner names the wait during drafting.
- A bare `/skill-creator` makes no backend call.
- A failed draft still opens the empty form.
- An empty pre-filled form can still be abandoned with escape.
- The creator lets the user pick the skill level: a global skill is visible but
  not "own"; a project skill lands in cwd and is removable.
- `/<skill> …` runs a normal turn with the skill's procedure forced into the
  model's copy of the message, not the stored transcript.
- A skill shows in the command menu; a built-in command wins a name clash.
- `/plan …` works out of the box on a fresh install.
- An unknown slash is still an error.
- `/skills-list` lists the profile's skills; lists shipped skills, tagged, when
  the user has none; own skills are untagged.
- `/skill-remove` removes the chosen skill; cancel keeps it; no own skills
  notifies.

## Compaction — `test_tui_compact.py`, `test_ui_compaction.py`

- `/compact` is offered in the slash menu and listed when `/` is typed.
- It writes a summary and shows it, and folds nothing until that summary is
  accepted: the review names how many messages would fold, the instruction the
  summary was written for, and whether the model's length budget cut it short.
- The review is inline and per session, not a modal: it stands in the message
  box's slot at the foot of the conversation it summarizes, which stays on
  screen behind it. A summary waiting in a background session flags that
  session's sidebar row and puts nothing on the current one; opening it reveals
  the prompt, and answering it there names that session.
- A summary longer than the prompt's share of the screen scrolls, and the rule
  says how far down it is. The prompt never takes more than half of what is
  left after the header and the footer.
- The comment box lives on the session, so a half-typed complaint survives a
  switch to another conversation and back.
- Enter accepts and the fold lands; the next turn runs on the summary.
- `r` asks for another summary with a line saying what was wrong with this one;
  that comment and the rejected text both reach the summarizer, and the next
  offer says which attempt it is. An empty comment is refused, not sent.
- `d` discards the summary and leaves the conversation as it was.
- Escape answers nothing and gives the message box back: the offer is kept, the
  sidebar still says so, and a bare `/compact` re-opens the
  same summary rather than writing a second one. An instruction after the
  command is a new brief, so it does write one.
- An instruction after the command steers the summary; a bare `/compact` asks
  for nothing in particular.
- The accepted summary is shown in full while it is being decided, and the fold
  is reported in the chat afterwards.
- The context meter stops showing the pre-fold measured number and falls back
  to an estimate — when the summary is accepted, not when the command is typed.
- A backend failure leaves the session untouched.
- Without a session there is nothing to compact; an empty session is left alone;
  compacting twice over says so.
- A session parked on a pending approval is not compacted.

## Node-local databases — `test_tui_dbcache.py`

- App tables, checkpoints and the RAG store all open from the node-local
  working dir, not `$HOME`.
- `$HOME` holds no live databases while running; the rest of the app dir stays
  in `$HOME`.
- On exit the databases are written back to `$HOME`; a session created in one
  run is in `$HOME` afterwards.
- The working dir is removed and the lease dropped on exit; a second run sees
  the first run's sessions.
- A sync timer is registered; the interval comes from settings; interval 0 means
  sync on exit only.
- Overlapping syncs are skipped; a sync failure is reported but not fatal.
- dbcache logging never reaches the terminal; it leaves its trail in the app
  dir; a first run creates the app dir it logs into.
- The disabled setting keeps DBs in `$HOME`; a second instance runs from `$HOME`
  and the user is told why; it still works end to end from `$HOME`.
- A working dir left behind by a crash is recovered on the next start, for every
  managed database.
- A corrupt home copy is reported at startup.
- On quit the app leaves the alt screen *first*, then prints the "copying
  databases home" message, before the copy starts, and announces the end of the
  wait; nothing is said when nothing is copied.
- During that wait the first Ctrl+C is answered, not obeyed; a second Ctrl+C
  aborts; the original handler is restored afterwards; nothing is guarded when
  nothing was said.

## Drafts — `test_tui_drafts.py`

- A draft does not follow the user into the next session.
- Returning to a session restores its draft; each session keeps its own; a
  multi-line draft comes back whole.
- A sent message leaves no draft behind.
- A new session starts with an empty entry.
- Closing and reopening a session keeps the draft.
- Deleting a session drops its draft.
- A parked `/…` draft brings its autocomplete menu back with it; a session with
  no draft does not inherit the previous menu.

## Undo in an editor — `test_ui_editor.py`

`ctrl+z` / `ctrl+y` in every buffer that takes typing: the message box, the
config editor and a profile's learnings.

- A run of typed characters undoes as one step, broken after whitespace, so one
  press takes back one word; a run of backspaces is one step too.
- A paste undoes whole, however many lines it turned out to be. So does a
  newline, a word deletion, `ctrl+u`, and anything that replaces a selection.
- Moving the cursor ends the run: what is typed after it is a step of its own.
- Undo restores the cursor to where the edit began, and leaves nothing selected.
- Redo goes forward as far as the last undo went; a new edit discards it.
- The stack is capped, and the oldest steps are dropped rather than the newest.
- A send forgets the stack: what has gone to the core is reached through the
  history below, not by undoing the box back into holding a copy of it.

## Message history — `test_ui_app.py`

↑/↓ in the message box walk this session's own messages, taken from the chat
log, so they come back with a conversation that is closed and reopened.

- ↑ steps back only from the top screen line of the box, ↓ forward only from the
  bottom one; anywhere else in a recalled multi-line message the arrows move the
  cursor as they always did.
- The draft in the box when the walk begins is stashed as the newest entry: ↓
  past the newest restores it, cursor and all.
- ↑ stops at the oldest rather than wrapping.
- The same message twice running is recalled once.
- A session with nothing sent in it leaves the arrows to the cursor.
- Editing ends the walk and drops the stash; so do escape, sending, leaving the
  box and switching session.
- The `/` menu still owns ↑/↓ while a command is being named.
- A recall does not go on the undo stack: the other arrow is the way back.

## Slash-command menu — `test_tui_autocomplete.py`

- Command usage counts are recorded and read back.
- `/` lists every command; matching is substring, not only prefix; `skill`
  narrows to the skill commands.
- A space after the name closes the menu: the command is settled, so ↑/↓ go
  back to the draft.
- No match hides the menu.
- Frequency sorts the most-used first.
- ↓/↑ move the selection; enter fills a partial command; tab also fills; enter
  on a complete command runs it.
- An unknown `/word` keeps the draft in place to be corrected; correcting it
  then runs.
- Skill descriptions are not parsed as markup.

## Navigating the entry — `test_tui_input_nav.py`, `test_tui.py`

- ↑ from a lower wrapped row stays inside the draft.
- ↑ from the top row browses the log.
- Arrows traverse the body of a slash command once the command name is settled.
- Arrows still pick a command while the name is being typed.
- ↓ then ↑ round-trips within the draft.
- Enter on `(new session)` asks for a profile, then opens the session; escaping
  the picker opens nothing.
- The chat entry only exists inside a session.
- Repeated "new session" reuses the empty one.
- Arrow keys leave the entry only at its edges; returning to chat resumes typing
  where it stopped.
- `c` opens the config editor, not offered in the chat column; escape closes
  without saving; the editor is prefilled with the current settings JSON; escape
  with changes asks, and saves on yes, discards on no; invalid JSON and invalid
  values show an error and keep it open.
- Copying text uses the clipboard manager.
- A wrapped draft grows the entry box, capped so the log stays visible.
- Enter sends; shift+enter, alt+enter and ctrl+j start a new line.
- ↑ moves between draft lines before leaving the entry.

## Selecting text in the chat — `test_tui_chat_selection.py`

Gated on the mouse being grabbed, which is off by default (see
specs-ui-replacement.md §4.1 item 5). With the mouse released the terminal's own
selection does this and these claims are satisfied by not interfering.

- Dragging across a message marks text.
- Marking a message does not jump focus to the entry.
- Marking the reasoning box does not collapse it.
- A marked selection copies via the clipboard manager (OSC 52).
- A plain click on a message still focuses the entry.
- A plain click on your own message offers the rewind dialog: click activates
  like Enter.
- A plain click toggles the reasoning box once, not twice.

## Reasoning effort — `test_tui_thinking.py`

- `/reasoning` opens a chooser with all four levels, each with its hint and none
  flagged unusable; the current level is starred and preselected; escape leaves
  the level alone; without a session it says so.
- A choice is stored on the session; new sessions start at the configured
  default; two sessions keep their own levels; an empty stored level follows the
  setting.
- The context bar shows the level for the open session, follows a change
  immediately, and hides with no session.
- "off" puts no level on the wire; a chosen level reaches the next decision
  request.
- The command is offered in the slash menu.

## Watchers column — `test_tui_watches.py`

- A watched log gets a box of its own, saying how long ago the log was last
  written.
- The boxes are the whole column; there are no subprocess history rows.
- A watch stays in the session it was made in; with no session open the column
  shows no watches.
- Watches survive a restart.
- Enter flashes the tail of the log as a toast, not a screen to dismiss.
- Peeking a job box shows its state; a job HPCA submitted peeks its output too.
- `d` removes the watch and its box but leaves the log file alone; `d` on an
  empty column does nothing.
- alt+↑/alt+↓ reorder boxes; the cursor follows the box it moved; the ends are
  silent; the arrangement outlives the 2s repaint; shift+arrows do the same for
  multiplexer users; an empty column has nothing to move.
- A column with boxes always has one highlighted: first paint lands on the top
  box; arriving by ←/→ finds the hotkeys; switching to a session with different
  boxes keeps a cursor; dropping a box holds the position, not the row; dropping
  the last box clamps; an empty column offers nothing.
- The cursor stays put while the clock ticks, the text still refreshes in place,
  and a new watch appears without losing the selection.
- A log going quiet raises no toast; a log that vanishes does; the first poll of
  a fresh watch is not news.

## Background processes and jobs — `test_tui_process_events.py`, `test_tui_jobs.py`

- A failed background script reaches the agent as an event.
- The event shows in the transcript: the user must see why the agent suddenly
  spoke.
- A foreground run produces no event, because the tool result already carries
  the failure.
- An event that arrives mid-turn is held until the turn ends — no concurrent
  `ainvoke` on one thread_id.
- Job tools are registered when Slurm is available and absent without it.
- A job state change from the sacct poll is written back.
- A submitted job gets no box of its own in the watchers column.

## Lag instrumentation — `test_tui_looplag.py`

- The lag probe is off unless `HPCA_LOOPLAG` is set; the env var turns it on;
  disabled means no task is ever started.
- An idle app blames nothing; a spike names the running step; a background turn
  counts; one step is named once; it survives being called before the app is built.
- A run leaves a report block behind; a disabled run leaves nothing.

## Context meter — `test_context_bar.py`

`render_bar` and `severity` are pure functions and are untouched. The widget
state they feed must survive:

- A "no reply yet" placeholder before the first measurement.
- A measured value replaces the placeholder.
- A measured value supersedes an estimate.
- Reset clears back to "no reply".
- A window arriving after the usage is applied.
- Speed is appended to the measured line, decimal for slow turns, waits for a
  measured fill, and is cleared by None or reset.
- Reasoning effort is shown before the first reply and appended after fill and
  speed; "off" is shown; None clears it.

## Carried over from deleted Textual-mechanics tests

- ESC-CR must decode as `alt+enter`, distinct from a bare `enter`; kitty
  sequences are untouched. (was `test_tui_termkeys.py`)
- A toast must never crash on arbitrary text, including text containing square
  brackets from LLM output. (was `test_tui_notify.py`)
- Columns survive a narrow terminal without collapsing or crashing; a wide
  terminal is unchanged. (was `test_tui_resize.py`)
