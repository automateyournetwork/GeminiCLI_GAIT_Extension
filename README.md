GAIT — Git for AI Tracking (Gemini-CLI Extension)

GAIT (Git for AI Tracking) brings Git-style version control to AI conversations.

This repository provides a Gemini-CLI extension that automatically tracks prompts, responses, context, and branching history using GAIT as a persistent backend.

Think of GAIT as version control for AI reasoning.

What This Is

This project integrates GAIT with Gemini-CLI using the Model Context Protocol (MCP).

Gemini-CLI acts as the interactive chat shell.
GAIT acts as the versioned context system.
MCP tools connect the two.

Once installed, every normal Gemini interaction is automatically tracked without changing how you chat.

No wrappers.
No forks.
No modified Gemini binaries.

What GAIT Tracks

GAIT records:

User prompts

Assistant responses

Conversation history as commits

Branches and merges

Reverts and resets

Pinned memory (curated context)

Optional token accounting

Optional remote sync

All data is stored locally in a .gait/ directory, similar to .git/.

Design Principles
Gemini-CLI Is the UI

Gemini-CLI remains responsible for:

Model execution

Authentication

Conversation flow

GAIT is responsible for:

Persistence

Versioning

Context control

History management

GAIT never wraps or replaces Gemini-CLI.

MCP-Native Architecture

GAIT is exposed as MCP tools.

This allows:

Clean slash commands

Structured tool calls

No stdout interference

Full compatibility with Gemini-CLI

Git-Like Mental Model

If you understand Git, GAIT will feel familiar.

Conversations are commits.
Conversation paths are branches.
Undo is revert.
Pinned memory is curated working context.
Remotes allow sharing.

Installation
Prerequisites

Gemini-CLI installed and working

Python 3.9 or newer

GAIT installed (pip install gait-ai)

Install the Extension

Clone this repository:

git clone https://github.com/your-org/gait-gemini-extension.git
cd gait-gemini-extension


Gemini-CLI automatically detects the extension via gemini-extension.json.

Getting Started
Start Gemini-CLI
gemini

Initialize GAIT

Change into a project directory (not filesystem root):

/gait:init


GAIT refuses to initialize at filesystem root for safety.

This creates a .gait/ directory that stores all history.

Normal Usage (Auto-Tracked)

Just chat normally.

Every assistant response is automatically recorded:

You: Explain CAP theorem
AI: The CAP theorem states that...


No extra commands required.

Branching Conversations

Create a new branch to explore ideas:

/gait:branch experiments
/gait:checkout experiments


Now you can diverge safely without affecting main.

Reverting (Undoing Mistakes)

Undo the most recent tracked turn:

/gait:revert


Revert multiple turns:

/gait:revert HEAD~2


Important notes:

This does not delete the visible Gemini-CLI transcript

It resets GAIT context going forward

Future responses will not include reverted content

Pinning Important Context

Pin the last meaningful answer into memory:

/gait:pin last


Pinned memory:

Is injected into future prompts

Persists across branches

Acts as curated working context

View pinned memory:

/gait:memory

Merging Conversation Branches

Merge a branch back into the current branch:

/gait:merge experiments --with-memory


This creates a merge commit and optionally merges pinned memory.

Remote Repositories (Optional)

GAIT supports syncing conversation history to a remote GAIT server.

Set a remote:

/gait:remote origin http://gaithub.example.com


Push:

/gait:push origin --owner alice --repo my-project


Pull:

/gait:pull origin --owner alice --repo my-project


This enables collaboration and shared AI workspaces.

Slash Commands Overview

Common commands:

/gait:init
/gait:status
/gait:branch NAME
/gait:checkout NAME
/gait:merge NAME
/gait:revert [TARGET]
/gait:pin last
/gait:memory
/gait:context
/gait:push
/gait:pull
/gait:clone


Slash commands perform operations and are not recorded as conversation turns.

Important Behavior Rules

Slash commands are not recorded as chat turns

Every normal assistant response is recorded

GAIT never writes to stdout

GAIT refuses unsafe operations

If GAIT is not initialized, you will be prompted to initialize

Who This Is For

GAIT is designed for:

Developers working with LLMs

Prompt engineers

AI researchers

Teams collaborating on AI workflows

Anyone who wants reproducible AI reasoning

Philosophy

AI conversations are stateful.

State deserves version control.

GAIT applies software engineering discipline to AI interaction.