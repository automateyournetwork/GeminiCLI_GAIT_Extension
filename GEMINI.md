GAIT Tracking Rules (MANDATORY)
0) Never write GAIT/MCP logs to STDOUT

All logs must go to STDERR only (MCP protocol owns STDOUT).

1) GAIT must exist in the working folder

If GAIT is not initialized in the current working folder, instruct the user to run:

gait_init()

GAIT must never be initialized at filesystem root (/, C:\, etc). If the user attempts it, refuse and tell them to cd into a project folder first.

2) Slash commands are tools, not chat

When the user types a slash command, do not answer normally. Translate it into tool calls and then provide a short confirmation.

Mappings:

/status → gait_status()

/branch NAME → gait_branch(name=NAME)

/checkout NAME → gait_checkout(name=NAME)

/revert [TARGET] → gait_revert(target=TARGET or "HEAD~1")

/pin last → gait_pin(last=true)

/memory → gait_memory()

/context → gait_context(full=false)

/push ... → gait_push(...)

/pull ... → gait_pull(...)

/clone ... → gait_clone(...)

Important: Slash command interactions are operational and must NOT be recorded via gait_record_turn().

3) Record every normal conversation turn

After every normal assistant response (non-slash-command), call:

gait_record_turn(user_text="<exact last user message>", assistant_text="<exact full assistant response>")

Rules:

user_text must be the user’s most recent message verbatim.

assistant_text must be your response verbatim.

Do not include extra meta text like “Recorded in GAIT” unless the tool call succeeded.

# LOOP GUARD (CRITICAL)

Never call gait_record_turn() in response to:
- Any tool output/result (including GAIT MCP tool results)
- Any message that contains JSON like {"ok": ...} or mentions "wrapper (gait MCP Server)"
- Any assistant message whose primary purpose is acknowledging a tool call (e.g. "gait committed", "recorded", "saved", "done")

Only call gait_record_turn() when the last user message is an actual human prompt/question AND it is not a /gait:* slash command.

If the user used a /gait:* command:
- Execute the tool call(s)
- Show the result
- DO NOT call gait_record_turn() for that exchange

4) Do not claim success unless it actually succeeded

Do not claim GAIT recorded a turn unless gait_record_turn returns { ok: true }.

If it fails, say:

GAIT is not initialized (or give the returned error),

and tell the user to run gait_init() in a project folder.

If you *DO CLAIM SUCCESS DO NOT RECORD YOUR RESPONSE IN GAIT AGAIN*. THIS *WILL* CAUSE INFINITE LOOPS.

5) Revert semantics

/revert does not delete the visible Gemini-CLI transcript.
It does reset GAIT HEAD so reverted turns are excluded from GAIT context going forward.

After /revert, continue the conversation using the reverted GAIT context.