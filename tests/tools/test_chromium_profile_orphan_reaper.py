from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.chromium_profile_reaper import (
    ChromiumOrphanFamily,
    ProcessSnapshot,
    default_process_snapshot_provider,
    default_remove_profile,
    discover_live_agent_browser_daemons,
    execute_chromium_orphan_reap,
    find_orphaned_chromium_families,
    make_jsonl_event_logger,
    make_revalidator_by_create_time,
    run_chromium_profile_reaper,
    terminate_chromium_family,
)


def _chrome(
    pid: int,
    *,
    ppid: int,
    started_at: float,
    profile: str,
    process_type: str | None = None,
) -> ProcessSnapshot:
    cmdline = [
        "/usr/bin/chromium",
        f"--user-data-dir={profile}",
    ]
    if process_type:
        cmdline.append(f"--type={process_type}")
    return ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        started_at=started_at,
        name="chrome",
        cmdline=tuple(cmdline),
    )


def test_classifica_familia_antiga_sem_daemon_como_orfa():
    profile = "/tmp/agent-browser-chrome-12345678-1234-4234-8234-123456789abc"
    processes = [
        _chrome(pid=100, ppid=1, started_at=100.0, profile=profile),
        _chrome(pid=101, ppid=100, started_at=101.0, profile=profile, process_type="renderer"),
    ]

    families = find_orphaned_chromium_families(
        processes,
        live_daemon_pids=set(),
        now=1_000.0,
        min_age_seconds=600,
    )

    assert len(families) == 1
    family = families[0]
    assert family.root_pid == 100
    assert family.profile == profile
    assert family.age_seconds == 900.0
    assert family.member_pids == (100, 101)


def test_dry_run_registra_orfao_sem_encerrar_ou_apagar_perfil():
    family = ChromiumOrphanFamily(
        root_pid=100,
        profile="/tmp/agent-browser-chrome-12345678-1234-4234-8234-123456789abc",
        age_seconds=900.0,
        member_pids=(100, 101),
    )
    events = []
    terminated = []
    removed = []

    result = execute_chromium_orphan_reap(
        [family],
        dry_run=True,
        terminate_family=lambda item: terminated.append(item.root_pid),
        profile_still_in_use=lambda _profile: False,
        remove_profile=lambda profile: removed.append(profile),
        log_event=events.append,
    )

    assert result.identified_root_pids == (100,)
    assert result.terminated_root_pids == ()
    assert terminated == []
    assert removed == []
    assert events == [
        {
            "event": "chromium_orphan_identified",
            "mode": "dry_run",
            "root_pid": 100,
            "age_seconds": 900.0,
            "profile": family.profile,
            "member_pids": [100, 101],
        }
    ]


def test_execucao_registra_resultado_e_remove_perfil_depois_do_encerramento():
    family = ChromiumOrphanFamily(
        root_pid=200,
        profile="/tmp/agent-browser-chrome-22345678-1234-4234-8234-123456789abc",
        age_seconds=1_200.0,
        member_pids=(200, 201),
    )
    order = []

    result = execute_chromium_orphan_reap(
        [family],
        dry_run=False,
        terminate_family=lambda item: order.append(("terminate", item.root_pid)),
        profile_still_in_use=lambda profile: order.append(("check", profile)) or False,
        remove_profile=lambda profile: order.append(("remove", profile)),
        log_event=lambda event: order.append(("log", event["event"])),
    )

    assert order == [
        ("log", "chromium_orphan_identified"),
        ("terminate", 200),
        ("log", "chromium_orphan_terminated"),
        ("check", family.profile),
        ("remove", family.profile),
        ("log", "chromium_orphan_profile_removed"),
    ]
    assert result.terminated_root_pids == (200,)
    assert result.removed_profiles == (family.profile,)


def test_encerramento_usa_sigterm_espera_cinco_segundos_e_escala_sobrevivente():
    family = ChromiumOrphanFamily(
        root_pid=300,
        profile="/tmp/agent-browser-chrome-32345678-1234-4234-8234-123456789abc",
        age_seconds=1_800.0,
        member_pids=(300, 301),
    )
    events = []

    class FakeProcess:
        def __init__(self, pid, survives):
            self.pid = pid
            self.survives = survives

        def terminate(self):
            events.append(("term", self.pid))

        def is_running(self):
            return self.survives

        def kill(self):
            events.append(("force", self.pid))

    processes = {300: FakeProcess(300, False), 301: FakeProcess(301, True)}

    terminate_chromium_family(
        family,
        process_factory=processes.__getitem__,
        sleep=lambda seconds: events.append(("sleep", seconds)),
        grace_seconds=5.0,
    )

    assert events == [
        ("term", 301),
        ("term", 300),
        ("sleep", 5.0),
        ("force", 301),
    ]


def test_descobre_daemon_vivo_por_sessao_ativa_e_valida_identidade(tmp_path):
    session_name = "h_abc123"
    socket_dir = tmp_path / f"agent-browser-{session_name}"
    socket_dir.mkdir()
    (socket_dir / f"{session_name}.pid").write_text("500", encoding="utf-8")
    daemon = ProcessSnapshot(
        pid=500,
        ppid=1,
        started_at=100.0,
        name="agent-browser-linux-x64",
        cmdline=("/opt/agent-browser-linux-x64",),
    )
    recycled = ProcessSnapshot(
        pid=501,
        ppid=1,
        started_at=100.0,
        name="python",
        cmdline=("python", "service.py"),
    )
    (socket_dir / "recycled.pid").write_text("501", encoding="utf-8")

    daemon_pids = discover_live_agent_browser_daemons(
        [daemon, recycled],
        active_sessions={"task": {"session_name": session_name}},
        socket_roots=[tmp_path],
    )

    assert daemon_pids == {500}


def test_execucao_pula_familia_que_falha_na_revalidacao_final():
    family = ChromiumOrphanFamily(
        root_pid=600,
        profile="/tmp/agent-browser-chrome-62345678-1234-4234-8234-123456789abc",
        age_seconds=1_800.0,
        member_pids=(600,),
    )
    events = []
    terminated = []

    result = execute_chromium_orphan_reap(
        [family],
        dry_run=False,
        revalidate_family=lambda _family: False,
        terminate_family=lambda item: terminated.append(item.root_pid),
        profile_still_in_use=lambda _profile: False,
        remove_profile=lambda _profile: None,
        log_event=events.append,
    )

    assert terminated == []
    assert result.skipped_root_pids == (600,)
    assert events[-1]["event"] == "chromium_orphan_revalidation_failed"


def test_orquestrador_em_dry_run_descobre_e_registra_sem_sinalizar(tmp_path):
    profile = "/tmp/agent-browser-chrome-72345678-1234-4234-8234-123456789abc"
    processes = [_chrome(pid=700, ppid=1, started_at=100.0, profile=profile)]
    events = []

    result = run_chromium_profile_reaper(
        dry_run=True,
        active_sessions={},
        min_age_seconds=600,
        now=1_000.0,
        process_snapshot_provider=lambda: processes,
        socket_roots=[tmp_path],
        log_event=events.append,
    )

    assert result.identified_root_pids == (700,)
    assert result.terminated_root_pids == ()
    assert events[0]["event"] == "chromium_orphan_scan_started"
    assert events[1]["event"] == "chromium_orphan_identified"
    assert events[-1]["event"] == "chromium_orphan_scan_completed"


def test_ignora_processo_que_nao_e_chromium_mesmo_com_perfil_valido():
    profile = "/tmp/agent-browser-chrome-82345678-1234-4234-8234-123456789abc"
    impostor = ProcessSnapshot(
        pid=800,
        ppid=1,
        started_at=100.0,
        name="python",
        cmdline=("python", f"--user-data-dir={profile}"),
    )

    families = find_orphaned_chromium_families(
        [impostor],
        live_daemon_pids=set(),
        now=1_000.0,
        min_age_seconds=600,
    )

    assert families == []


class _FakePsutilProcess:
    def __init__(self, pid: int, create_time: float):
        self.pid = pid
        self._create_time = create_time

    def create_time(self):
        return self._create_time


def test_revalidador_aprova_familia_quando_create_time_permanece_igual():
    profile = "/tmp/agent-browser-chrome-92345678-1234-4234-8234-123456789abc"
    snapshot = ProcessSnapshot(
        pid=900, ppid=1, started_at=1_000.0, name="chrome",
        cmdline=("/usr/bin/chromium", f"--user-data-dir={profile}"),
    )
    family = ChromiumOrphanFamily(
        root_pid=900, profile=profile, age_seconds=1_800.0, member_pids=(900,),
    )
    revalidate = make_revalidator_by_create_time(
        {900: snapshot},
        process_factory=lambda pid: _FakePsutilProcess(pid, 1_000.0),
    )

    assert revalidate(family) is True


def test_revalidador_rejeita_familia_quando_create_time_mudou():
    profile = "/tmp/agent-browser-chrome-a2345678-1234-4234-8234-123456789abc"
    snapshot = ProcessSnapshot(
        pid=1_000, ppid=1, started_at=1_000.0, name="chrome",
        cmdline=("/usr/bin/chromium", f"--user-data-dir={profile}"),
    )
    family = ChromiumOrphanFamily(
        root_pid=1_000, profile=profile, age_seconds=1_800.0, member_pids=(1_000,),
    )
    revalidate = make_revalidator_by_create_time(
        {1_000: snapshot},
        process_factory=lambda pid: _FakePsutilProcess(pid, 5_000.0),
    )

    assert revalidate(family) is False


def test_revalidador_falha_fechado_quando_inspecao_lanca_excecao():
    profile = "/tmp/agent-browser-chrome-b2345678-1234-4234-8234-123456789abc"
    snapshot = ProcessSnapshot(
        pid=1_100, ppid=1, started_at=1_000.0, name="chrome",
        cmdline=("/usr/bin/chromium", f"--user-data-dir={profile}"),
    )
    family = ChromiumOrphanFamily(
        root_pid=1_100, profile=profile, age_seconds=1_800.0, member_pids=(1_100,),
    )

    def raise_no_such_process(pid):
        raise RuntimeError("simulated psutil.NoSuchProcess")

    revalidate = make_revalidator_by_create_time(
        {1_100: snapshot}, process_factory=raise_no_such_process,
    )

    assert revalidate(family) is False


def test_revalidador_falha_fechado_quando_snapshot_do_pid_esta_ausente():
    """Sem baseline capturado, não há como validar — comportamento fail-closed."""
    profile = "/tmp/agent-browser-chrome-c2345678-1234-4234-8234-123456789abc"
    family = ChromiumOrphanFamily(
        root_pid=1_200, profile=profile, age_seconds=1_800.0, member_pids=(1_200,),
    )
    revalidate = make_revalidator_by_create_time(
        {},  # baseline vazio
        process_factory=lambda pid: _FakePsutilProcess(pid, 1_000.0),
    )

    assert revalidate(family) is False


def test_revalidador_valida_todos_os_membros_da_familia():
    """Se qualquer PID da família teve create_time alterado, rejeita a família inteira."""
    profile = "/tmp/agent-browser-chrome-d2345678-1234-4234-8234-123456789abc"
    snap_root = ProcessSnapshot(
        pid=1_300, ppid=1, started_at=1_000.0, name="chrome",
        cmdline=("/usr/bin/chromium", f"--user-data-dir={profile}"),
    )
    snap_child = ProcessSnapshot(
        pid=1_301, ppid=1_300, started_at=1_001.0, name="chrome",
        cmdline=("/usr/bin/chromium", "--type=renderer", f"--user-data-dir={profile}"),
    )
    family = ChromiumOrphanFamily(
        root_pid=1_300, profile=profile, age_seconds=1_800.0,
        member_pids=(1_300, 1_301),
    )

    def process_factory(pid):
        if pid == 1_300:
            return _FakePsutilProcess(pid, 1_000.0)  # root: match
        return _FakePsutilProcess(pid, 9_000.0)  # child: mismatch → PID reciclado

    revalidate = make_revalidator_by_create_time(
        {1_300: snap_root, 1_301: snap_child}, process_factory=process_factory,
    )

    assert revalidate(family) is False


def test_provider_psutil_converte_process_iter_para_snapshots(monkeypatch):
    import tools.chromium_profile_reaper as reaper_module

    class _FakeIterProc:
        def __init__(self, info):
            self.info = info

    class _FakePsutilModule:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        AccessDenied = type("AccessDenied", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (Exception,), {})

        @staticmethod
        def process_iter(attrs):
            return [
                _FakeIterProc({
                    "pid": 42, "ppid": 1, "create_time": 1_000.5,
                    "name": "chromium", "cmdline": ["/usr/bin/chromium", "--headless"],
                }),
                _FakeIterProc({
                    "pid": 43, "ppid": 42, "create_time": 1_001.0,
                    "name": "chromium", "cmdline": ["/usr/bin/chromium", "--type=renderer"],
                }),
            ]

    monkeypatch.setattr(reaper_module, "psutil", _FakePsutilModule, raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutilModule)

    snapshots = default_process_snapshot_provider()

    assert len(snapshots) == 2
    root = snapshots[0]
    assert root.pid == 42
    assert root.ppid == 1
    assert root.started_at == 1_000.5
    assert root.name == "chromium"
    assert root.cmdline == ("/usr/bin/chromium", "--headless")


def test_provider_psutil_tolera_processos_que_evaporam_durante_iteracao(monkeypatch):
    import sys
    import tools.chromium_profile_reaper as reaper_module

    class _FakePsutilModule:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        AccessDenied = type("AccessDenied", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (Exception,), {})

    class _ExplodingProc:
        @property
        def info(self):
            raise _FakePsutilModule.NoSuchProcess()

    class _OKProc:
        info = {
            "pid": 99, "ppid": 1, "create_time": 1_000.0,
            "name": "chrome", "cmdline": ["/usr/bin/chrome"],
        }

    _FakePsutilModule.process_iter = staticmethod(
        lambda attrs: [_ExplodingProc(), _OKProc()]
    )

    monkeypatch.setattr(reaper_module, "psutil", _FakePsutilModule, raising=False)
    monkeypatch.setitem(sys.modules, "psutil", _FakePsutilModule)

    snapshots = default_process_snapshot_provider()

    assert [s.pid for s in snapshots] == [99]


def test_logger_jsonl_escreve_uma_linha_por_evento_com_timestamp(tmp_path):
    log_path = tmp_path / "sub" / "chromium-orphan-reaper.jsonl"
    log_event = make_jsonl_event_logger(
        log_path, now_iso=lambda: "2026-09-08T12:00:00+00:00",
    )

    log_event({"event": "chromium_orphan_scan_started", "process_count": 42})
    log_event({"event": "chromium_orphan_identified", "root_pid": 100})

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["timestamp"] == "2026-09-08T12:00:00+00:00"
    assert first["event"] == "chromium_orphan_scan_started"
    assert first["process_count"] == 42
    assert second["event"] == "chromium_orphan_identified"
    assert second["root_pid"] == 100


def test_logger_jsonl_engole_erros_de_io_sem_propagar(tmp_path):
    """Reaper nunca falha um scan por causa do log de auditoria."""
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o500)
    log_path = read_only_dir / "chromium-orphan-reaper.jsonl"
    log_event = make_jsonl_event_logger(log_path)

    # Não deve levantar exceção mesmo com diretório read-only.
    log_event({"event": "chromium_orphan_scan_started"})

    read_only_dir.chmod(0o700)  # cleanup pra tmp_path fixture


def test_orquestrador_registra_todos_os_seis_eventos_em_dry_run(tmp_path):
    """Cobre os 6 eventos JSONL: scan_started, identified, scan_completed via dry_run.
    (revalidation_failed, terminated, profile_removed cobertos por outros testes.)"""
    profile = "/tmp/agent-browser-chrome-e2345678-1234-4234-8234-123456789abc"
    processes = [_chrome(pid=1_400, ppid=1, started_at=100.0, profile=profile)]
    log_path = tmp_path / "reaper.jsonl"
    log_event = make_jsonl_event_logger(
        log_path, now_iso=lambda: "2026-09-08T12:00:00+00:00",
    )

    run_chromium_profile_reaper(
        dry_run=True,
        active_sessions={},
        min_age_seconds=600,
        now=1_000.0,
        process_snapshot_provider=lambda: processes,
        socket_roots=[tmp_path],
        log_event=log_event,
    )

    lines = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    event_names = [entry["event"] for entry in lines]
    assert event_names == [
        "chromium_orphan_scan_started",
        "chromium_orphan_identified",
        "chromium_orphan_scan_completed",
    ]
    # Todos com timestamp injetado.
    assert all(entry["timestamp"] == "2026-09-08T12:00:00+00:00" for entry in lines)


def test_execute_mode_ainda_levanta_runtimeerror_ate_aprovacao_explicita(tmp_path):
    """A guarda de ativação não pode ter sido removida por engano em nenhuma edição."""
    processes = [_chrome(
        pid=1_500, ppid=1, started_at=100.0,
        profile="/tmp/agent-browser-chrome-f2345678-1234-4234-8234-123456789abc",
    )]

    with pytest.raises(RuntimeError, match="execute mode requires explicit system dependencies"):
        run_chromium_profile_reaper(
            dry_run=False,
            active_sessions={},
            min_age_seconds=600,
            now=1_000.0,
            process_snapshot_provider=lambda: processes,
            socket_roots=[tmp_path],
            log_event=lambda _event: None,
        )


def test_remove_profile_recusa_caminho_fora_do_tempdir(tmp_path):
    """Defesa em profundidade: profile fora de /tmp nunca deve ser deletado."""
    import tempfile as _tempfile

    outsider = tmp_path / "agent-browser-chrome-12345678-1234-4234-8234-123456789abc"
    outsider.mkdir()
    marker = outsider / "canary.txt"
    marker.write_text("preserved")
    assert Path(_tempfile.gettempdir()) != tmp_path  # fixture não é o tempdir do SO

    default_remove_profile(str(outsider))

    assert marker.exists(), "profile fora do tempdir do sistema não pode ser removido"


def test_remove_profile_recusa_nome_fora_do_padrao(tmp_path, monkeypatch):
    """Nome que não bate com o regex agent-browser-chrome-<uuid> nunca é apagado."""
    import tempfile as _tempfile

    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))
    from tools import chromium_profile_reaper as reaper_module
    monkeypatch.setattr(reaper_module, "tempfile", _tempfile)

    fake = tmp_path / "some-random-folder"
    fake.mkdir()
    (fake / "keep-me.txt").write_text("safe")

    default_remove_profile(str(fake))

    assert fake.exists(), "diretório com nome fora do padrão deve ser preservado"


def test_remove_profile_remove_diretorio_valido(tmp_path, monkeypatch):
    import tempfile as _tempfile

    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))
    from tools import chromium_profile_reaper as reaper_module
    monkeypatch.setattr(reaper_module, "tempfile", _tempfile)

    profile_dir = tmp_path / "agent-browser-chrome-12345678-1234-4234-8234-123456789abc"
    profile_dir.mkdir()
    (profile_dir / "cache").mkdir()
    (profile_dir / "cache" / "data.bin").write_bytes(b"junk")

    default_remove_profile(str(profile_dir))

    assert not profile_dir.exists()


def test_remove_profile_com_symlink_apaga_link_e_preserva_target(tmp_path, monkeypatch):
    """Se profile for symlink, remove o link e nunca o target — evita TOCTOU."""
    import tempfile as _tempfile

    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))
    from tools import chromium_profile_reaper as reaper_module
    monkeypatch.setattr(reaper_module, "tempfile", _tempfile)

    real_target = tmp_path / "important-data"
    real_target.mkdir()
    canary = real_target / "keep.txt"
    canary.write_text("do not delete")

    symlink_profile = tmp_path / "agent-browser-chrome-12345678-1234-4234-8234-123456789abc"
    try:
        symlink_profile.symlink_to(real_target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks não suportados neste ambiente")

    default_remove_profile(str(symlink_profile))

    assert not symlink_profile.exists() and not symlink_profile.is_symlink()
    assert canary.exists() and canary.read_text() == "do not delete"
