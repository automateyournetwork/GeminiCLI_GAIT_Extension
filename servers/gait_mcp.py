#!/usr/bin/env python3
"""
MCP Server: GAIT (Git for AI Tracking) for Gemini-CLI.

Gemini-CLI is the chat UI.
This server provides GAIT repo/memory/remote tools plus gait_record_turn()
so every Gemini turn can be persisted automatically.

Design goals:
- STDIO MCP: NEVER write to stdout except protocol (FastMCP handles this)
- Log to STDERR only
- Be resilient when GAIT is not initialized yet (return structured errors)
- Enforce: DO NOT init at filesystem root
- Provide a GAIT-native /revert via gait_revert() (reset semantics)
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Logging (stderr only)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("GaitMCP")

# ---------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# ---------------------------------------------------------------------
# GAIT
# ---------------------------------------------------------------------
from gait.repo import GaitRepo
from gait.schema import Turn
from gait.tokens import count_turn_tokens
from gait.objects import short_oid

# If your package exposes remote ops similar to your CLI, wire them here.
# IMPORTANT: ensure these functions do not print to stdout.
# from gait.remote import push as remote_push, pull as remote_pull, clone as remote_clone
# from gait.remote import remote_set as remote_set, remote_get as remote_get, remote_list as remote_list

mcp = FastMCP("GAIT MCP")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _err(msg: str, *, detail: str = "", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error": msg}
    if detail:
        out["detail"] = detail
    out.update(extra)
    return out


def _try_repo() -> Tuple[Optional[GaitRepo], Optional[Dict[str, Any]]]:
    """
    Discover GAIT repo from CWD. Return (repo, None) on success,
    (None, structured_error_dict) on failure.
    """
    try:
        return (GaitRepo.discover(), None)
    except Exception as e:
        return (None, _err("GAIT repo not found. Run /gait:init inside a project folder.", detail=str(e)))


def _require_repo() -> GaitRepo:
    repo, err = _try_repo()
    if err or repo is None:
        # FastMCP will serialize exception; we prefer structured returns in tools,
        # but internal helpers can still raise.
        raise RuntimeError(err["error"] if err else "GAIT repo not found")
    return repo


def _is_filesystem_root(p: Path) -> bool:
    """
    True if p is filesystem root.
    Works for POSIX '/' and Windows 'C:\\' (anchor-based).
    """
    p = p.resolve()
    return str(p) == str(Path(p.anchor).resolve())


def _walk_first_parent(repo: GaitRepo, start: str, limit: int) -> List[Dict[str, Any]]:
    """
    Walk commit history along first parent.
    Requires repo.get_commit(commit_id) returning dict with 'parents' (list).
    """
    out: List[Dict[str, Any]] = []
    cid = start
    seen = set()

    while cid and cid not in seen and len(out) < max(0, int(limit)):
        seen.add(cid)
        c = repo.get_commit(cid)
        out.append(
            {
                "id": cid,
                "message": c.get("message", ""),
                "created_at": c.get("created_at", ""),
                "parents": c.get("parents") or [],
                "turn_ids": c.get("turn_ids") or [],
            }
        )
        parents = c.get("parents") or []
        cid = parents[0] if parents else ""

    return out


def _resolve_commit_prefix_from_head(repo: GaitRepo, head: str, prefix: str) -> str:
    """
    Resolve a short commit prefix by scanning first-parent from HEAD.
    """
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("empty commit prefix")

    cid = head
    seen = set()
    while cid and cid not in seen:
        seen.add(cid)
        if cid.startswith(prefix):
            return cid
        c = repo.get_commit(cid)
        parents = c.get("parents") or []
        cid = parents[0] if parents else ""

    raise ValueError(f"Unknown commit or prefix: {prefix}")


def _resolve_revert_target(repo: GaitRepo, target: str) -> str:
    """
    Supports:
      - "" or "HEAD~1" -> one step back
      - "HEAD~N"
      - short/full commit id prefix (scanned from HEAD)
    Returns the commit id to reset the branch ref to.
    """
    t = (target or "").strip()
    head = repo.head_commit_id() or ""
    if not head:
        raise ValueError("No HEAD commit to revert")

    # default
    if not t:
        t = "HEAD~1"

    # HEAD~N
    m = re.fullmatch(r"HEAD~(\d+)", t.upper())
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError("HEAD~N must be >= 1")

        cid = head
        for _ in range(n):
            c = repo.get_commit(cid)
            parents = c.get("parents") or []
            cid = parents[0] if parents else ""
            if not cid:
                break
        if not cid:
            raise ValueError(f"Cannot revert: {t} is beyond history")
        return cid

    # commit id / prefix
    return _resolve_commit_prefix_from_head(repo, head, t)


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@mcp.tool()
def gait_status() -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    return {
        "ok": True,
        "root": str(repo.root),
        "branch": repo.current_branch(),
        "head": repo.head_commit_id() or "",
    }


@mcp.tool()
def gait_init(path: str = ".") -> Dict[str, Any]:
    root = Path(path).resolve()

    # HARD RULE: refuse init at filesystem root
    if _is_filesystem_root(root):
        return _err(
            "Refusing to initialize GAIT at filesystem root. cd into a working folder first.",
            path=str(root),
        )

    repo = GaitRepo(root=root)
    repo.init()
    return {"ok": True, "root": str(repo.root), "gait_dir": str(repo.gait_dir)}


@mcp.tool()
def gait_branch(
    name: str,
    from_commit: Optional[str] = None,
    inherit_memory: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    try:
        repo.create_branch(name, from_commit=from_commit, inherit_memory=inherit_memory)
        return {"ok": True, "created": True, "branch": name}
    except FileExistsError:
        if not force:
            return {"ok": True, "created": False, "branch": name, "note": "already exists (use force=true to reset)"}

        target = from_commit if from_commit is not None else repo.head_commit_id()
        repo.write_ref(name, target or "")

        if inherit_memory:
            # keep memory mapping aligned to current branch memory
            repo.write_memory_ref(repo.read_memory_ref(repo.current_branch()), name)

        return {"ok": True, "created": False, "reset": True, "branch": name, "head": target or ""}


@mcp.tool()
def gait_checkout(name: str) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    repo.checkout(name)
    return {"ok": True, "branch": repo.current_branch(), "head": repo.head_commit_id() or ""}


@mcp.tool()
def gait_log(limit: int = 20) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    head = repo.head_commit_id() or ""
    commits = _walk_first_parent(repo, head, limit)

    out = [
        {"commit": short_oid(c["id"]), "message": c.get("message", ""), "created_at": c.get("created_at", "")}
        for c in commits
    ]
    return {
        "ok": True,
        "branch": repo.current_branch(),
        "head": short_oid(head) if head else "",
        "commits": out,
    }


@mcp.tool()
def gait_show(commit: str = "") -> Dict[str, Any]:
    """
    Show commit details by:
      - commit id/prefix
      - or default HEAD when empty
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    head = repo.head_commit_id() or ""
    if not head:
        return _err("No HEAD commit to show")

    cid = head if not commit.strip() else _resolve_commit_prefix_from_head(repo, head, commit.strip())
    c = repo.get_commit(cid)

    # You can expand this if your schema supports embedding turns here.
    return {
        "ok": True,
        "commit": cid,
        "short": short_oid(cid),
        "message": c.get("message", ""),
        "created_at": c.get("created_at", ""),
        "parents": c.get("parents") or [],
        "turn_ids": c.get("turn_ids") or [],
    }


@mcp.tool()
def gait_memory() -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    manifest = repo.get_memory()
    items = []
    for i, it in enumerate(manifest.items, start=1):
        items.append(
            {
                "index": i,
                "turn": short_oid(it.turn_id),
                "commit": short_oid(it.commit_id),
                "note": it.note,
            }
        )
    return {"ok": True, "branch": repo.current_branch(), "pinned": len(items), "items": items}


@mcp.tool()
def gait_context(full: bool = False) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    return {"ok": True, "bundle": repo.build_context_bundle(full=full)}


@mcp.tool()
def gait_pin(commit: Optional[str] = None, last: bool = True, note: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.pin_commit(commit, last=last, note=note or "")
    return {"ok": True, "memory_id": mem_id}


@mcp.tool()
def gait_unpin(index: int) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.unpin_index(index)
    return {"ok": True, "unpinned": index, "memory_id": mem_id}


@mcp.tool()
def gait_record_turn(
    user_text: str,
    assistant_text: str,
    note: str = "gemini-cli",
    use_memory_snapshot: bool = True,
) -> Dict[str, Any]:
    """
    Called by Gemini after every response to persist the turn.

    IMPORTANT:
    - If GAIT is not initialized, return a structured error so GEMINI.md can prompt user to /gait:init.
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    context: Dict[str, Any] = {}

    if use_memory_snapshot:
        try:
            context["pinned_context"] = repo.build_context_bundle(full=False)
        except Exception as e:
            context["pinned_context_error"] = str(e)

    tokens = count_turn_tokens(user_text=user_text, assistant_text=assistant_text)

    turn = Turn.v0(
        user_text=user_text,
        assistant_text=assistant_text,
        context=context,
        tools={},
        model={"provider": "gemini-cli"},
        tokens=tokens,
        visibility="private",
    )

    _, commit_id = repo.record_turn(turn, message=note)
    return {"ok": True, "commit": short_oid(commit_id), "branch": repo.current_branch()}


@mcp.tool()
def gait_revert(target: str = "HEAD~1") -> Dict[str, Any]:
    """
    Reset semantics revert:
    - Moves the current branch ref to the resolved target (default HEAD~1).
    - This changes GAIT context going forward.
    - It does NOT delete Gemini-CLI transcript (cannot).
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    branch = repo.current_branch()
    head_before = repo.head_commit_id() or ""
    if not head_before:
        return _err("No HEAD commit to revert")

    try:
        new_head = _resolve_revert_target(repo, target)
    except Exception as e:
        return _err("Failed to resolve revert target", detail=str(e), target=target)

    try:
        repo.write_ref(branch, new_head)
    except Exception as e:
        return _err("Failed to update branch ref", detail=str(e), branch=branch)

    return {
        "ok": True,
        "branch": branch,
        "head_before": short_oid(head_before),
        "head_after": short_oid(new_head),
        "note": "GAIT context is now reverted going forward (Gemini-CLI transcript is unchanged).",
    }


# ---------------------------------------------------------------------
# Remote tools (wire to your gait.remote module when ready)
# ---------------------------------------------------------------------


@mcp.tool()
def gait_push(remote: str, owner: str, repo: str, branch: str = "", token: str = "") -> Dict[str, Any]:
    # TODO: replace stub with your existing GAIT remote push implementation.
    # result = remote_push(remote=remote, owner=owner, repo=repo, branch=branch, token=token)
    # return {"ok": True, **result}
    return _err("gait_push not wired yet (connect to gait.remote.push)")


@mcp.tool()
def gait_pull(remote: str, owner: str, repo: str, branch: str = "", token: str = "") -> Dict[str, Any]:
    # TODO: replace stub with your existing GAIT remote pull implementation.
    return _err("gait_pull not wired yet (connect to gait.remote.pull)")


@mcp.tool()
def gait_clone(remote: str, owner: str, repo: str, path: str, branch: str = "", token: str = "") -> Dict[str, Any]:
    # TODO: replace stub with your existing GAIT remote clone implementation.
    return _err("gait_clone not wired yet (connect to gait.remote.clone)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
