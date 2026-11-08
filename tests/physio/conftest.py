"""Synthetic fixtures for the physiological engineering layer."""

from __future__ import annotations

import pytest
from physio_testkit import device_payload as _device_payload


@pytest.fixture
def device_payload():
    return _device_payload()
