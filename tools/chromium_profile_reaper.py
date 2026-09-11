from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# The kernel exposes process start time in seconds since the epoch through
# ``psutil.Process.create_time()``. Small clock skews (a few hundred ms) are
# expected between what /proc/<pid>/stat reports and the ``started_at`` we
# captured during the scan; a strict equality check would produce false
# revalidation failures. 0.5s is well above observed jitter and well below
# the timescale on which PIDs are recycled onto a completely different
# process, so it keeps the PID-reuse guard tight without being brittle.
_CREATE_TIME_TOLERANCE_SECONDS = 0.5

_PROFILE_NAME_RE = re.compile(
    r"^agent-browser-chrome-[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int
    started_at: float
    name: str
    cmdline: tuple[str, ...]


@dataclass(frozen=True)
class ChromiumOrphanFamily:
    root_pid: int
    profile: str
    age_seconds: float
    member_pids: tuple[int, ...]


@dataclass(frozen=True)
class ChromiumOrphanReapResult:
    identified_root_pids: tuple[int, ...]
    terminated_root_pids: tuple[int, ...]
    removed_profiles: tuple[str, ...]
    skipped_root_pids: tuple[int, ...]


def _temporary_agent_browser_profile(process: ProcessSnapshot) -> str | None:
    prefix = "--user-data-dir="
    raw = next((arg[len(prefix) :] for arg in process.cmdline if arg.startswith(prefix)), None)
    if not raw:
        return None
    path = Path(raw)
    if path.parent != Path(tempfile.gettempdir()):
        return None
    if not _PROFILE_NAME_RE.fullmatch(path.name):
        return None
    return str(path)


def _is_chromium_process(process: ProcessSnapshot) -> bool:
    executable = Path(process.cmdline[0]).name.lower() if process.cmdline else ""
    process_name = process.name.lower()
    allowed_names = {
        "chrome",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "chrome-headless-shell",
        "headless_shell",
    }
    return process_name in allowed_names or executable in allowed_names


def _is_browser_root(process: ProcessSnapshot) -> bool:
    return _is_chromium_process(process) and not any(
        arg.startswith("--type=") for arg in process.cmdline
    )


def _is_agent_browser_daemon(process: ProcessSnapshot) -> bool:
    identity = " ".join((process.name, *process.cmdline)).lower()
    return "agent-browser" in identity


def discover_live_agent_browser_daemons(
    processes: Iterable[ProcessSnapshot],
    *,
    active_sessions: Mapping[str, Mapping[str, Any]],
    socket_roots: Iterable[Path],
) -> set[int]:
    by_pid = {process.pid: process for process in processes}
    active_names = {
        str(info.get("session_name"))
        for info in active_sessions.values()
        if info.get("session_name")
    }
    pid_files = set()
    for root in socket_roots:
        if not root.exists():
            continue
        pid_files.update(root.rglob("*.pid"))
        for session_name in active_names:
            pid_files.add(
                root / f"agent-browser-{session_name}" / f"{session_name}.pid"
            )

    daemon_pids = set()
    for pid_file in pid_files:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        process = by_pid.get(pid)
        if process is not None and _is_agent_browser_daemon(process):
            daemon_pids.add(pid)
    return daemon_pids


def _has_live_daemon_ancestor(
    process: ProcessSnapshot,
    by_pid: dict[int, ProcessSnapshot],
    live_daemon_pids: set[int],
) -> bool:
    seen = set()
    parent_pid = process.ppid
    while parent_pid and parent_pid not in seen:
        if parent_pid in live_daemon_pids:
            return True
        seen.add(parent_pid)
        parent = by_pid.get(parent_pid)
        if parent is None:
            break
        parent_pid = parent.ppid
    return False


def find_orphaned_chromium_families(
    processes: Iterable[ProcessSnapshot],
    *,
    live_daemon_pids: set[int],
    now: float,
    min_age_seconds: float,
) -> list[ChromiumOrphanFamily]:
    snapshots = list(processes)
    by_pid = {process.pid: process for process in snapshots}
    members_by_profile: dict[str, list[int]] = {}
    for process in snapshots:
        profile = _temporary_agent_browser_profile(process)
        if profile:
            members_by_profile.setdefault(profile, []).append(process.pid)

    families = []
    for process in snapshots:
        profile = _temporary_agent_browser_profile(process)
        if not profile or not _is_browser_root(process):
            continue
        age_seconds = max(0.0, now - process.started_at)
        if age_seconds <= min_age_seconds:
            continue
        if _has_live_daemon_ancestor(process, by_pid, live_daemon_pids):
            continue
        families.append(
            ChromiumOrphanFamily(
                root_pid=process.pid,
                profile=profile,
                age_seconds=age_seconds,
                member_pids=tuple(sorted(members_by_profile[profile])),
            )
        )
    return sorted(families, key=lambda family: family.root_pid)


def execute_chromium_orphan_reap(
    families: Iterable[ChromiumOrphanFamily],
    *,
    dry_run: bool,
    revalidate_family: Callable[[ChromiumOrphanFamily], bool] = lambda _family: True,
    terminate_family: Callable[[ChromiumOrphanFamily], None],
    profile_still_in_use: Callable[[str], bool],
    remove_profile: Callable[[str], None],
    log_event: Callable[[dict], None],
) -> ChromiumOrphanReapResult:
    identified = []
    terminated = []
    removed = []
    skipped = []
    for family in families:
        identified.append(family.root_pid)
        log_event(
            {
                "event": "chromium_orphan_identified",
                "mode": "dry_run" if dry_run else "execute",
                "root_pid": family.root_pid,
                "age_seconds": family.age_seconds,
                "profile": family.profile,
                "member_pids": list(family.member_pids),
            }
        )
        if dry_run:
            continue
        if not revalidate_family(family):
            skipped.append(family.root_pid)
            log_event(
                {
                    "event": "chromium_orphan_revalidation_failed",
                    "root_pid": family.root_pid,
                    "age_seconds": family.age_seconds,
                    "profile": family.profile,
                }
            )
            continue
        terminate_family(family)
        terminated.append(family.root_pid)
        log_event(
            {
                "event": "chromium_orphan_terminated",
                "root_pid": family.root_pid,
                "age_seconds": family.age_seconds,
                "profile": family.profile,
                "member_pids": list(family.member_pids),
            }
        )
        if not profile_still_in_use(family.profile):
            remove_profile(family.profile)
            removed.append(family.profile)
            log_event(
                {
                    "event": "chromium_orphan_profile_removed",
                    "root_pid": family.root_pid,
                    "age_seconds": family.age_seconds,
                    "profile": family.profile,
                }
            )
    return ChromiumOrphanReapResult(
        identified_root_pids=tuple(identified),
        terminated_root_pids=tuple(terminated),
        removed_profiles=tuple(removed),
        skipped_root_pids=tuple(skipped),
    )


def terminate_chromium_family(
    family: ChromiumOrphanFamily,
    *,
    process_factory: Callable[[int], Any],
    sleep: Callable[[float], None] = time.sleep,
    grace_seconds: float = 5.0,
) -> None:
    ordered_pids = [pid for pid in family.member_pids if pid != family.root_pid]
    ordered_pids.append(family.root_pid)
    processes = []
    for pid in ordered_pids:
        try:
            process = process_factory(pid)
            process.terminate()
            processes.append(process)
        except Exception:
            continue
    sleep(grace_seconds)
    for process in processes:
        try:
            if process.is_running():
                process.kill()
        except Exception:
            continue


def default_process_snapshot_provider() -> list[ProcessSnapshot]:
    """Enumerate live processes via ``psutil`` into ``ProcessSnapshot`` tuples.

    Injected as the ``process_snapshot_provider`` for real (non-test) runs.
    Processes that vanish or become inaccessible mid-iteration are silently
    skipped — the scanner is best-effort and any process that raced away is,
    by definition, not an orphan we need to reap.
    """
    import psutil

    snapshots: list[ProcessSnapshot] = []
    for proc in psutil.process_iter(
        ["pid", "ppid", "create_time", "name", "cmdline"]
    ):
        try:
            info = proc.info
            pid = info.get("pid")
            ppid = info.get("ppid")
            create_time = info.get("create_time")
            if pid is None or ppid is None or create_time is None:
                continue
            snapshots.append(
                ProcessSnapshot(
                    pid=int(pid),
                    ppid=int(ppid),
                    started_at=float(create_time),
                    name=str(info.get("name") or ""),
                    cmdline=tuple(info.get("cmdline") or ()),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return snapshots


def make_revalidator_by_create_time(
    snapshots_by_pid: Mapping[int, ProcessSnapshot],
    *,
    process_factory: Callable[[int], Any] | None = None,
    tolerance_seconds: float = _CREATE_TIME_TOLERANCE_SECONDS,
) -> Callable[[ChromiumOrphanFamily], bool]:
    """Build a revalidator that re-checks kernel create_time immediately before signalling.

    Between the scan and the signal, the kernel may recycle a PID onto an
    unrelated process. Comparing the ``create_time`` captured during the scan
    against the live value catches that: a mismatch means we are looking at a
    different process now and must not signal it.

    Fails closed on any inspection error (``NoSuchProcess``, ``AccessDenied``,
    permission failure, missing baseline snapshot). A leaked orphan is strictly
    preferable to killing a stranger.
    """
    if process_factory is None:
        import psutil
        process_factory = psutil.Process

    def revalidate(family: ChromiumOrphanFamily) -> bool:
        for pid in family.member_pids:
            expected = snapshots_by_pid.get(pid)
            if expected is None:
                return False
            try:
                proc = process_factory(pid)
                live_create_time = proc.create_time()
            except Exception:
                return False
            try:
                delta = abs(float(live_create_time) - float(expected.started_at))
            except (TypeError, ValueError):
                return False
            if delta > tolerance_seconds:
                return False
        return True

    return revalidate


def default_socket_roots() -> list[Path]:
    """Directories where ``agent-browser`` daemons keep per-session socket + .pid files.

    ``discover_live_agent_browser_daemons`` walks these roots for ``*.pid``
    files and cross-checks the referenced PID against live snapshots, so it's
    safe to enumerate broadly — non-agent-browser pid files are ignored.
    """
    roots: list[Path] = [Path(tempfile.gettempdir())]
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        roots.append(Path(xdg_runtime))
    home = os.path.expanduser("~")
    if home:
        roots.append(Path(home) / ".agent-browser")
    return roots


def default_jsonl_log_path() -> Path:
    """Standard location for the reaper's JSONL event log."""
    return Path(os.path.expanduser("~")) / ".hermes" / "logs" / "chromium-orphan-reaper.jsonl"


def make_jsonl_event_logger(
    log_path: Path,
    *,
    now_iso: Callable[[], str] | None = None,
) -> Callable[[dict], None]:
    """Return a ``log_event`` callable that appends one JSON object per line.

    The parent directory is created lazily on the first write. Timestamps are
    UTC ISO-8601 with microsecond precision. Any write error is swallowed —
    the reaper never fails a scan because the audit log is unavailable.
    """
    if now_iso is None:
        now_iso = lambda: datetime.now(timezone.utc).isoformat()

    def log_event(event: dict) -> None:
        record = {"timestamp": now_iso(), **event}
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        except OSError:
            return

    return log_event


def default_remove_profile(profile: str) -> None:
    """Remove a Chromium temp profile directory.

    Only deletes paths matching the ``agent-browser-chrome-<uuid>`` naming
    schema under the system temp directory. If the path is a symlink, the
    link itself is removed and its target is left untouched — a hostile
    process cannot trick the janitor into wiping an unrelated directory by
    swapping the profile out for a symlink between scan and cleanup.
    """
    path = Path(profile)
    if path.parent != Path(tempfile.gettempdir()):
        return
    if not _PROFILE_NAME_RE.fullmatch(path.name):
        return
    if path.is_symlink():
        try:
            path.unlink()
        except OSError:
            pass
        return
    shutil.rmtree(path, ignore_errors=True)


def default_profile_still_in_use(profile: str) -> bool:
    """True if any live process still references the profile via ``--user-data-dir``.

    A survivor with the same profile means the terminate step didn't fully
    clear the family — leaving the directory in place lets the survivor
    continue to function and gives the next scan a chance to catch it.
    """
    for snapshot in default_process_snapshot_provider():
        if _temporary_agent_browser_profile(snapshot) == profile:
            return True
    return False


def run_chromium_profile_reaper(
    *,
    dry_run: bool,
    active_sessions: Mapping[str, Mapping[str, Any]],
    min_age_seconds: float = 600.0,
    now: float | None = None,
    process_snapshot_provider: Callable[[], Sequence[ProcessSnapshot]],
    socket_roots: Iterable[Path],
    log_event: Callable[[dict], None],
) -> ChromiumOrphanReapResult:
    current_time = time.time() if now is None else now
    snapshots = list(process_snapshot_provider())
    log_event(
        {
            "event": "chromium_orphan_scan_started",
            "mode": "dry_run" if dry_run else "execute",
            "process_count": len(snapshots),
            "min_age_seconds": min_age_seconds,
        }
    )
    daemon_pids = discover_live_agent_browser_daemons(
        snapshots,
        active_sessions=active_sessions,
        socket_roots=socket_roots,
    )
    families = find_orphaned_chromium_families(
        snapshots,
        live_daemon_pids=daemon_pids,
        now=current_time,
        min_age_seconds=min_age_seconds,
    )
    # Execute mode is intentionally still gated on an explicit human review
    # of the wired-up implementation. When the operator authorizes activation,
    # this guard is lifted and the caller supplies real terminate/remove/
    # revalidate closures — none of which are hard-coded here so a stray
    # ``dry_run=False`` can never signal a live process.
    if not dry_run:
        raise RuntimeError("execute mode requires explicit system dependencies")
    result = execute_chromium_orphan_reap(
        families,
        dry_run=True,
        terminate_family=lambda _family: None,
        profile_still_in_use=lambda _profile: True,
        remove_profile=lambda _profile: None,
        log_event=log_event,
    )
    log_event(
        {
            "event": "chromium_orphan_scan_completed",
            "mode": "dry_run",
            "identified_count": len(result.identified_root_pids),
            "terminated_count": 0,
        }
    )
    return result
