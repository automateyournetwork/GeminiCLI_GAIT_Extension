# GAIT Tracking Rules (MANDATORY)

1) If GAIT is not initialized in the current working folder, ask the user to run:
   - gait_init()
   AND refuse to initialize if the current folder is a filesystem root.

2) After EVERY assistant response (no exceptions), call:
   - gait_record_turn(user_text="<the user's latest message>", assistant_text="<your full response>")

3) When the user types slash commands, translate them to tool calls:
   - /status -> gait_status()
   - /branch NAME -> gait_branch(name=NAME)
   - /checkout NAME -> gait_checkout(name=NAME)
   - /revert TARGET -> gait_revert(target=TARGET)
   - /pin last -> gait_pin(last=true)
   - /memory -> gait_memory()
   - /context -> gait_context(full=false)
   - /push ... -> gait_push(...)
   - /pull ... -> gait_pull(...)
   - /clone ... -> gait_clone(...)

Do not claim GAIT recorded a turn unless the gait_record_turn tool call succeeds.