"""Regression: auto-follow resolves stored channel ids to channel objects.

The voice auto-follow map (``_voice_follow_members``) stores channel **ids**
(ints, from ``after.channel.id`` in ``on_voice_state_update``). ``join_voice_channel``
and the ``target.id`` consumers require a real channel object, so a raw int
leaking out of ``_resolve_voice_follow_target`` crashed with
``AttributeError: 'int' object has no attribute 'guild'`` every time an allowed
user moved channels while auto-follow was enabled.

These tests pin the read-site contract: the resolved target is always a channel
object (or None), never a bare id.
"""

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