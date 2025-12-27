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
- Provide GAIT-native revert/reset semantics (optionally also reset memory)
- Provide remote add/list/get + push/fetch/pull/clone + repo create
"""

from __future__ import annotations

import logging
import os
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
# GAIT core
# ---------------------------------------------------------------------
from gait.repo import GaitRepo
from gait.schema import Turn
from gait.tokens import count_turn_tokens
from gait.objects import short_oid
from gait.log import walk_commits

# ---------------------------------------------------------------------
# GAIT remote (your real implementation)
# ---------------------------------------------------------------------
from gait.remote import (
    RemoteSpec,
    remote_add,
    remote_get,
    remote_list,
    push as remote_push,
    fetch as remote_fetch,
    pull as remote_pull,
    clone_into,
    create_repo as remote_create_repo,
)

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
    try:
        return (GaitRepo.discover(), None)
    except Exception as e:
        return (None, _err("GAIT repo not found. Run /gait:init inside a project folder.", detail=str(e)))


def _is_filesystem_root(p: Path) -> bool:
    p = p.resolve()
    return str(p) == str(Path(p.anchor).resolve())


def _get_gaithub_token(provided: str = "") -> Optional[str]:
    t = (provided or "").strip()
    if t:
        return t
    env = os.environ.get("GAITHUB_TOKEN", "").strip()
    return env or None


def _require_gaithub_token(provided: str = "") -> str:
    tok = _get_gaithub_token(provided)
    if not tok:
        raise RuntimeError("GAITHUB_TOKEN is not set (and token was not provided)")
    return tok


def _remote_spec(repo: GaitRepo, remote: str, owner: str, repo_name: str) -> RemoteSpec:
    base_url = remote_get(repo, remote)
    return RemoteSpec(base_url=base_url, owner=owner, repo=repo_name, name=remote)


def _resolve_commit_prefix_from_head(repo: GaitRepo, head: str, prefix: str) -> str:
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
    Returns the commit id to reset the branch ref to (or "" for empty)
    """
    t = (target or "").strip()
    head = repo.head_commit_id() or ""
    if not head:
        raise ValueError("No HEAD commit to revert")

    if not t:
        t = "HEAD~1"

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

        if cid == "":
            # beyond history => empty branch
            return ""
        return cid

    # otherwise interpret as prefix/full hash
    return _resolve_commit_prefix_from_head(repo, head, t)


def _safe_tool(fn):
    """
    Decorator: converts uncaught exceptions into structured tool errors.
    Keeps server stable and avoids protocol stdout.
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.exception("tool failed: %s", fn.__name__)
            return _err(f"{fn.__name__} failed", detail=str(e))
    return wrapper


# ---------------------------------------------------------------------
# Core repo tools
# ---------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def gait_status() -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    return {"ok": True, "root": str(repo.root), "branch": repo.current_branch(), "head": repo.head_commit_id() or ""}


@mcp.tool()
@_safe_tool
def gait_init(path: str = ".") -> Dict[str, Any]:
    root = Path(path).resolve()
    if _is_filesystem_root(root):
        return _err("Refusing to initialize GAIT at filesystem root. cd into a working folder first.", path=str(root))
    repo = GaitRepo(root=root)
    repo.init()
    return {"ok": True, "root": str(repo.root), "gait_dir": str(repo.gait_dir)}


@mcp.tool()
@_safe_tool
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
            repo.write_memory_ref(repo.read_memory_ref(repo.current_branch()), name)

        return {"ok": True, "created": False, "reset": True, "branch": name, "head": target or ""}


@mcp.tool()
@_safe_tool
def gait_checkout(name: str) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    repo.checkout(name)
    return {"ok": True, "branch": repo.current_branch(), "head": repo.head_commit_id() or ""}


@mcp.tool()
@_safe_tool
def gait_delete_branch(name: str, force: bool = False) -> Dict[str, Any]:
    """
    Mirrors cmd_delete behavior:
    - refuse delete 'main' unless force
    - refuse delete current branch unless force (and switch to main first)
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    b = name.strip()
    if b == "main" and not force:
        return _err("Refusing to delete 'main' without force=true", branch=b)

    cur = repo.current_branch()
    if b == cur and not force:
        return _err("Refusing to delete current branch without force=true", branch=b)

    ref = repo.refs_dir / b
    mem = repo.memory_refs_dir / b

    if not ref.exists():
        return {"ok": True, "deleted": False, "branch": b, "note": "no such branch"}

    if b == cur and force:
        if (repo.refs_dir / "main").exists():
            repo.checkout("main")
        else:
            return _err("Cannot force-delete current branch: no 'main' branch exists", branch=b)

    ref.unlink()
    if mem.exists():
        mem.unlink()

    return {"ok": True, "deleted": True, "branch": b, "current_branch": repo.current_branch()}


@mcp.tool()
@_safe_tool
def gait_merge(source: str, message: str = "", with_memory: bool = False) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    merge_id = repo.merge(source, message=message or "", with_memory=with_memory)
    out = {"ok": True, "merged": source, "branch": repo.current_branch(), "head": short_oid(merge_id)}
    if with_memory:
        out["memory"] = repo.read_memory_ref(repo.current_branch())
    return out


@mcp.tool()
@_safe_tool
def gait_log(limit: int = 20) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    commits = []
    for c in walk_commits(repo, limit=limit):
        cid = c["_id"]
        parents = c.get("parents") or []
        commits.append(
            {
                "commit": short_oid(cid),
                "id": cid,
                "created_at": c.get("created_at") or "",
                "kind": c.get("kind") or "",
                "message": c.get("message") or "",
                "parents": [short_oid(x) for x in parents],
                "turns": len(c.get("turn_ids") or []),
                "merge": len(parents) > 1,
            }
        )
    return {"ok": True, "branch": repo.current_branch(), "commits": commits}


@mcp.tool()
@_safe_tool
def gait_show(commit: str = "HEAD") -> Dict[str, Any]:
    """
    Return commit details + attached turns (user/assistant text).
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    head = repo.head_commit_id() or ""
    if not head:
        return _err("No commits in this branch yet")

    cid = head if commit in ("", "HEAD", "@") else _resolve_commit_prefix_from_head(repo, head, commit)
    c = repo.get_commit(cid)

    turn_ids = c.get("turn_ids") or []
    turns = []
    for tid in turn_ids:
        t = repo.get_turn(tid)
        turns.append(
            {
                "turn_id": tid,
                "user": (t.get("user") or {}).get("text", ""),
                "assistant": (t.get("assistant") or {}).get("text", ""),
            }
        )

    return {
        "ok": True,
        "commit": cid,
        "short": short_oid(cid),
        "created_at": c.get("created_at") or "",
        "kind": c.get("kind") or "",
        "message": c.get("message") or "",
        "parents": c.get("parents") or [],
        "turns": turns,
    }


# ---------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def gait_memory() -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    manifest = repo.get_memory()
    items = []
    for i, it in enumerate(manifest.items, start=1):
        items.append({"index": i, "turn": short_oid(it.turn_id), "commit": short_oid(it.commit_id), "note": it.note})
    return {"ok": True, "branch": repo.current_branch(), "pinned": len(items), "items": items}


@mcp.tool()
@_safe_tool
def gait_context(full: bool = False) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    return {"ok": True, "bundle": repo.build_context_bundle(full=full)}


@mcp.tool()
@_safe_tool
def gait_pin(commit: Optional[str] = None, last: bool = True, note: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.pin_commit(commit, last=last, note=note or "")
    return {"ok": True, "memory_id": mem_id}


@mcp.tool()
@_safe_tool
def gait_unpin(index: int) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.unpin_index(index)
    return {"ok": True, "unpinned": index, "memory_id": mem_id}


# ---------------------------------------------------------------------
# Turn recording (auto tracking)
# ---------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def gait_record_turn(
    user_text: str,
    assistant_text: str,
    note: str = "gemini-cli",
    use_memory_snapshot: bool = True,
) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------
# Revert / reset (matches your CLI cmd_revert)
# ---------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def gait_revert(target: str = "HEAD~1", also_memory: bool = False) -> Dict[str, Any]:
    """
    GAIT-native revert = reset branch HEAD (default: first parent of HEAD).
    Optionally also rewinds memory (like cmd_revert --also-memory).
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    branch = repo.current_branch()
    head_before = repo.head_commit_id() or ""
    if not head_before:
        return _err("Nothing to revert (branch has no commits).", branch=branch)

    # Resolve target -> commit id or "" (empty)
    new_head = _resolve_revert_target(repo, target)

    # Use your repo.reset_branch() if available (it resolves/validates)
    if new_head == "":
        repo.write_ref(branch, "")
        head_after = ""
    else:
        resolved = repo.reset_branch(new_head)
        head_after = resolved

    out: Dict[str, Any] = {
        "ok": True,
        "branch": branch,
        "head_before": short_oid(head_before),
        "head_after": short_oid(head_after) if head_after else "(empty)",
        "note": "GAIT history is now rewound. Gemini-CLI transcript cannot be erased, but GAIT context going forward uses the reverted history.",
    }

    if also_memory:
        old_mem = repo.read_memory_ref(branch)
        new_mem = repo.reset_memory_to_commit(branch, repo.head_commit_id())
        out["memory_before"] = old_mem
        out["memory_after"] = new_mem

    return out


# ---------------------------------------------------------------------
# Remote tools (wired to your gait.remote)
# ---------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def gait_remote_add(name: str, url: str) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    remote_add(repo, name, url)
    return {"ok": True, "remote": name, "url": url}


@mcp.tool()
@_safe_tool
def gait_remote_list(verbose: bool = True) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    rems = remote_list(repo)
    if verbose:
        return {"ok": True, "remotes": rems}
    return {"ok": True, "remotes": sorted(list(rems.keys()))}


@mcp.tool()
@_safe_tool
def gait_remote_get(name: str = "origin") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    url = remote_get(repo, name)
    return {"ok": True, "remote": name, "url": url}


@mcp.tool()
@_safe_tool
def gait_repo_create(remote: str, owner: str, repo_name: str, token: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _require_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)
    remote_create_repo(spec, token=tok)
    return {"ok": True, "created": f"{owner}/{repo_name}", "remote": remote}


@mcp.tool()
@_safe_tool
def gait_push(remote: str, owner: str, repo_name: str, branch: str = "", token: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _require_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)

    # match cmd_push behavior: if remote says repo not initialized, create then retry
    try:
        remote_push(repo, spec, token=tok, branch=branch or None)
    except RuntimeError as e:
        msg = str(e)
        if "Repo not initialized for this owner" in msg:
            remote_create_repo(spec, token=tok)
            remote_push(repo, spec, token=tok, branch=branch or None)
        else:
            raise

    return {
        "ok": True,
        "pushed": branch or repo.current_branch(),
        "remote": remote,
        "owner": owner,
        "repo": repo_name,
    }


@mcp.tool()
@_safe_tool
def gait_fetch(remote: str, owner: str, repo_name: str, token: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _get_gaithub_token(token)  # fetch can be anonymous depending on your server
    spec = _remote_spec(repo, remote, owner, repo_name)
    heads, mems = remote_fetch(repo, spec, token=tok)
    return {"ok": True, "remote": remote, "owner": owner, "repo": repo_name, "heads": len(heads), "memory": len(mems)}


@mcp.tool()
@_safe_tool
def gait_pull(
    remote: str,
    owner: str,
    repo_name: str,
    branch: str = "",
    with_memory: bool = False,
    token: str = "",
) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _get_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)

    merge_id = remote_pull(
        repo,
        spec,
        token=tok,
        branch=branch or repo.current_branch(),
        with_memory=with_memory,
    )

    out: Dict[str, Any] = {
        "ok": True,
        "pulled": f"{remote}/{branch or repo.current_branch()}",
        "into": repo.current_branch(),
        "head": merge_id,
    }
    if with_memory:
        out["memory"] = repo.read_memory_ref(repo.current_branch())
    return out


@mcp.tool()
@_safe_tool
def gait_clone(
    url: str,
    owner: str,
    repo_name: str,
    path: str,
    remote: str = "origin",
    branch: str = "main",
    token: str = "",
) -> Dict[str, Any]:
    tok = _get_gaithub_token(token)
    dest = Path(path).resolve()
    spec = RemoteSpec(base_url=url, owner=owner, repo=repo_name, name=remote)
    clone_into(dest, spec, token=tok, branch=branch)
    return {"ok": True, "cloned": f"{owner}/{repo_name}", "into": str(dest), "branch": branch, "remote": remote}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
