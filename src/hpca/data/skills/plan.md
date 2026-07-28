---
name: plan
description: Plan a change before building it — grill the user to a shared
  understanding, then write specs.md, an implementation plan a fresh session can
  build from.
triggers: [plan, specs]
---

1. **Grill first.** Follow the `grillme` skill exactly (fetch it with `read_skill`,
   name `grillme`): one question at a time, your recommended answer with each,
   facts looked up yourself, every decision left to the user. Write nothing until
   they confirm you have reached a shared understanding.
2. **Then write `specs.md`** in the working directory: what is being built and why,
   every decision settled during the grilling with its reasoning, the files to
   touch, the steps in order, how to verify each one, and what was ruled out.
   Detailed enough to implement from scratch in a fresh session that saw none of
   the conversation — leave no open questions in it.
3. **Stop there.** Implementing is a separate session; say the plan is ready.
