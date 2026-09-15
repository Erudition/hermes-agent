"""Regression: auto-follow resolves stored channel ids to channel objects.

The voice auto-follow map (``_voice_follow_members``) stores channel **ids**
(ints, from ``after.channel.id`` in ``on_voice_state_update``). ``join_voice_channel``
and the ``target.id`` consumers require a real channel object, so a raw int
leaking out of ``_resolve_voice_follow_target`` crashed with
``AttributeError: 'int' object has no attribute 'guild'`` every time an allowed
user moved channels while auto-follow was enabled.

These tests pin the read-site contract: the resolved target is always a channel
object (or None), never a bare id. Further tests pin the lock contract: the
auto-follow task must NOT hold the guild voice lock when it calls into
``join_voice_channel`` / ``leave_voice_channel`` — both acquire that lock
themselves, and ``asyncio.Lock`` is not reentrant, so an outer hold
self-deadlocks the task silently (follow never happens, no error logged).
"""

import asyncio
from types import SimpleNamespace


class _FakeChannel:
    def __init__(self, channel_id, guild_id):
        self.id = channel_id
        self.guild = SimpleNamespace(id=guild_id)


class _FakeClient:
    """Minimal discord client: get_channel resolves ids or returns None."""

    def __init__(self, channels=None):
        self.channels = dict(channels or {})
        self.resolved = []

    def get_channel(self, channel_id):
        self.resolved.append(channel_id)
        return self.channels.get(channel_id)


def _make_adapter(client):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._voice_follow_members = {}
    adapter._client = client
    return adapter


def test_resolve_voice_follow_target_returns_channel_object_not_id():
    """A stored int id resolves to a channel object the join path can use."""
    client = _FakeClient({555: _FakeChannel(555, 111)})
    adapter = _make_adapter(client)
    adapter._voice_follow_members = {111: {222: 555}}

    target = adapter._resolve_voice_follow_target(111)

    # The exact attributes consumed downstream: .guild (join_voice_channel)
    # and .id (target_id / text channel binding).
    assert target is not None
    assert target.guild.id == 111
    assert target.id == 555
    assert client.resolved == [555]


def test_resolve_voice_follow_target_latest_occupied_member_wins():
    """Dict order preserved: the most recent occupant decides the target."""
    client = _FakeClient({555: _FakeChannel(555, 111), 777: _FakeChannel(777, 111)})
    adapter = _make_adapter(client)
    adapter._voice_follow_members = {111: {222: 555, 333: 777}}

    target = adapter._resolve_voice_follow_target(111)

    assert target is not None
    assert target.id == 777


def test_resolve_voice_follow_target_none_when_unresolvable_or_out_of_voice():
    """None contract: gone channel and everyone-left both mean no target."""
    client = _FakeClient()  # no channels -> every resolution misses
    adapter = _make_adapter(client)

    # Channel no longer exists in the client cache -> treated as no target.
    adapter._voice_follow_members = {111: {222: 555}}
    assert adapter._resolve_voice_follow_target(111) is None

    # Member left voice (stored None) / unknown guild -> no target.
    adapter._voice_follow_members = {111: {222: None}}
    assert adapter._resolve_voice_follow_target(111) is None
    assert adapter._resolve_voice_follow_target(999) is None


def test_resolve_voice_follow_target_no_client_resolves_to_none():
    """Stored id with no client -> None, never the raw id (auto-leave path)."""
    adapter = _make_adapter(None)  # disconnected / pre-connect state
    adapter._voice_follow_members = {111: {222: 555}}

    target = adapter._resolve_voice_follow_target(111)

    assert target is None


def _make_task_adapter(client):
    """Adapter primed for _voice_auto_follow_task: one tracked member in VC 555."""
    adapter = _make_adapter(client)
    adapter._voice_follow_manual_guilds = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_follow_pending = {}
    adapter._voice_text_channels = {}
    adapter._voice_auto_follow_leave_delay = 0
    return adapter


def _make_lock_stub(adapter, recorder):
    """Stub join/leave that re-acquires the guild voice lock, exactly as the
    real join_voice_channel/leave_voice_channel do. Without this re-acquire
    the test cannot detect the task holding the lock across the call."""

    async def stub(target_or_guild_id, *args, **kwargs):
        guild_id = (
            target_or_guild_id.guild.id
            if hasattr(target_or_guild_id, "guild")
            else target_or_guild_id
        )
        recorder.append("entered")
        async with adapter._voice_locks.setdefault(guild_id, asyncio.Lock()):
            recorder.append("acquired")

    return stub


def test_auto_follow_task_does_not_hold_voice_lock_when_joining(monkeypatch):
    """Follow on move must complete: task may not self-deadlock on the lock."""
    client = _FakeClient({555: _FakeChannel(555, 111)})
    adapter = _make_task_adapter(client)
    adapter._voice_follow_members = {111: {222: 555}}
    monkeypatch.setattr(
        adapter, "_synthetic_voice_source",
        lambda guild_id, text_channel_id: {"platform": "discord"},
    )

    recorder = []
    monkeypatch.setattr(adapter, "join_voice_channel", _make_lock_stub(adapter, recorder))

    async def _run():
        await asyncio.wait_for(adapter._voice_auto_follow_task(111), timeout=3)

    asyncio.run(_run())

    assert recorder == ["entered", "acquired"]


def test_auto_follow_task_does_not_hold_voice_lock_when_leaving(monkeypatch):
    """Auto-leave must complete: same no-outer-lock contract, real _maybe_voice_auto_leave."""
    adapter = _make_task_adapter(None)
    adapter._voice_follow_members = {111: {}}
    adapter._voice_auto_follow_leave_delay = 1
    monkeypatch.setattr(adapter, "is_in_voice_channel", lambda guild_id: True)
    monkeypatch.setattr(adapter, "_reset_voice_timeout", lambda guild_id: None)

    recorder = []
    monkeypatch.setattr(adapter, "leave_voice_channel", _make_lock_stub(adapter, recorder))

    async def _run():
        await asyncio.wait_for(adapter._voice_auto_follow_task(111), timeout=5)

    asyncio.run(_run())

    assert recorder == ["entered", "acquired"]


# ---------------------------------------------------------------------------
# Channel moves must rebuild the voice transport, not ``move_to`` it.
#
# discord.py's ``VoiceClient.move_to`` re-handshakes the voice websocket but
# the UDP capture transport dies with the old connection (the socket reader
# ends up selecting on a closed fd; no RTP is ever delivered again).
# Observed live: SPEAKING events keep arriving after the move while capture is
# permanently silent — the bot follows but can never hear again. The follow
# path must therefore fully disconnect and reconnect, so the fresh-join path
# reinstalls the receiver on a brand-new transport.
# ---------------------------------------------------------------------------


class _FakeVoiceConnection:
    def __init__(self):
        self.secret_key = b"\x01" * 32
        self.ssrc = 42
        self.hook = None
        self.listeners = []

    def add_socket_listener(self, callback):
        self.listeners.append(callback)

    def remove_socket_listener(self, callback):
        if callback in self.listeners:
            self.listeners.remove(callback)


class _FakeVoiceClient:
    def __init__(self, channel_id):
        self.channel = SimpleNamespace(id=channel_id)
        self._connection = _FakeVoiceConnection()
        self.moved_to = None
        self.disconnected = False

    def is_connected(self):
        return not self.disconnected

    async def move_to(self, channel):
        self.moved_to = channel

    async def disconnect(self):
        self.disconnected = True


class _ConnectableChannel(_FakeChannel):
    def __init__(self, channel_id, guild_id, vc):
        super().__init__(channel_id, guild_id)
        self._vc = vc

    async def connect(self):
        return self._vc


def _make_join_adapter(client, existing_vc):
    adapter = _make_adapter(client)
    adapter._voice_locks = {}
    adapter._voice_clients = {111: existing_vc}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._allowed_user_ids = {"222"}
    return adapter


def test_join_voice_channel_reconnects_and_rebuilds_receiver_on_move(monkeypatch):
    """A follow across channels must reconnect (fresh transport + receiver),
    not ``move_to`` (which kills UDP capture permanently)."""
    import plugins.platforms.discord.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "DISCORD_AVAILABLE", True)

    old_vc = _FakeVoiceClient(555)
    old_receiver = adapter_module.VoiceReceiver(old_vc, allowed_user_ids={"222"})
    old_receiver.start()
    old_receiver.map_ssrc(30498, 329095202335621122)

    new_vc = _FakeVoiceClient(777)
    target = _ConnectableChannel(777, 111, new_vc)

    adapter = _make_join_adapter(_FakeClient(), old_vc)
    adapter._voice_receivers[111] = old_receiver
    monkeypatch.setattr(adapter, "_reset_voice_timeout", lambda guild_id: None)

    async def _run():
        assert await adapter.join_voice_channel(target) is True
        task = adapter._voice_listen_tasks.get(111)
        assert task is not None
        await asyncio.sleep(0)  # let the listen task take its first slice
        assert not task.done()

    asyncio.run(_run())

    assert old_vc.disconnected is True
    assert old_vc.moved_to is None
    assert adapter._voice_clients[111] is new_vc

    new_receiver = adapter._voice_receivers[111]
    assert new_receiver is not old_receiver
    assert new_receiver._running is True
    assert new_receiver._vc is new_vc
    assert new_vc._connection.listeners == [new_receiver._on_packet]
    assert new_receiver._ssrc_to_user.get(30498) == 329095202335621122


def test_join_voice_channel_same_channel_is_idempotent(monkeypatch):
    """Already in the target channel: no teardown, no reconnect, receiver kept."""
    import plugins.platforms.discord.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "DISCORD_AVAILABLE", True)

    vc = _FakeVoiceClient(555)
    receiver = adapter_module.VoiceReceiver(vc, allowed_user_ids={"222"})
    receiver.start()

    adapter = _make_join_adapter(_FakeClient(), vc)
    adapter._voice_receivers[111] = receiver
    monkeypatch.setattr(adapter, "_reset_voice_timeout", lambda guild_id: None)

    async def _run():
        assert await adapter.join_voice_channel(_FakeChannel(555, 111)) is True

    asyncio.run(_run())

    assert vc.disconnected is False
    assert vc.moved_to is None
    assert adapter._voice_receivers[111] is receiver
    assert receiver._running is True