import pytest

from fastlane.world.db import World
from fastlane.world.seed import build_world


@pytest.fixture(scope="session")
def world_path(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world") / "world.db")


@pytest.fixture
def world(world_path):
    w = World.fresh(world_path)
    yield w
    w.close()
