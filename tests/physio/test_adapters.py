"""#96 adapters: device/service/offline produce the same internal contract."""

from __future__ import annotations

import hashlib

import pytest
from physio_testkit import ts

from physio.adapters import (
    MOCK_ONLY,
    NOT_CONNECTED,
    DeviceEventAdapter,
    OfflineSessionAdapter,
    ServiceEventAdapter,
)
from physio.schema import QUALITY_DEGRADED, SOURCE_OFFLINE, SchemaError

pytestmark = pytest.mark.synthetic


def signature(event):
    return {
        (m.metric, m.value, m.unit, m.valid, m.status) for m in event.measurements
    }


def test_device_and_offline_semantics_are_equivalent(device_payload, tmp_path):
    device = DeviceEventAdapter().parse_event(device_payload)

    csv_path = tmp_path / "session.csv"
    rows = [
        "sequence,measured_at,received_at,phase,metric,value,unit",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,hr,72,bpm",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,spo2,0.97,fraction",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,skin_temp,33.5,°C",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,rr,0,breaths/min",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,sbp,0,mmHg",
        f"0,{device_payload['measured_at']},{device_payload['received_at']},"
        "during,dbp,80,mmHg",
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    session = OfflineSessionAdapter().from_csv(
        csv_path,
        session_id="sess-demo-1",
        subject_ref="subj-pseudo-01",
        device_ref="chair-mock-01",
    )
    offline = session.events[0]
    assert offline.source == SOURCE_OFFLINE
    assert offline.measured_at == device.measured_at
    assert offline.received_at == device.received_at
    assert signature(offline) == signature(device)
    assert offline.quality == device.quality == QUALITY_DEGRADED
    assert offline.protocol is not None and offline.protocol.program_id is None


def test_service_and_device_share_measurement_semantics(device_payload):
    device = DeviceEventAdapter().parse_event(device_payload)
    service = ServiceEventAdapter().parse_event(
        {**device_payload, "device_id": device_payload["device_ref"], "user_id": "subj-pseudo-01"}
    )
    assert service.source == "service"
    assert signature(service) == signature(device)
    assert service.measured_at == device.measured_at


def test_device_adapter_is_honest_about_not_connected():
    capabilities = DeviceEventAdapter().capabilities()
    assert capabilities.status == NOT_CONNECTED
    assert capabilities.protocol == MOCK_ONLY
    assert capabilities.realtime is True


def test_offline_adapter_is_not_a_realtime_transport():
    capabilities = OfflineSessionAdapter().capabilities()
    assert capabilities.realtime is False
    assert "research import only" in " ".join(capabilities.notes)


def test_csv_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "session.csv"
    path.write_text("sequence,measured_at,received_at,phase,metric,value,unit\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="sha256 mismatch"):
        OfflineSessionAdapter().from_csv(
            path,
            session_id="s1",
            subject_ref="p1",
            device_ref="d1",
            expected_sha256="0" * 64,
        )


def test_csv_import_records_hash_and_license(tmp_path):
    path = tmp_path / "session.csv"
    path.write_text(
        "sequence,measured_at,received_at,phase,metric,value,unit\n"
        "0,2026-11-07T10:00:00+00:00,2026-11-07T10:00:02+00:00,during,hr,70,bpm\n",
        encoding="utf-8",
    )
    session = OfflineSessionAdapter().from_csv(
        path, session_id="s1", subject_ref="p1", device_ref="d1"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert session.provenance.raw_sha256 == digest
    assert session.provenance.data_license == "PRIVATE_NOT_FOR_COMMIT"


def test_offline_json_session_roundtrip(device_payload, tmp_path):
    device = DeviceEventAdapter().parse_event(device_payload)
    from physio.schema import assemble_session

    session = assemble_session(
        device.session_id, device.subject_ref, device.device_ref, [device]
    )
    path = tmp_path / "session.json"
    path.write_text(__import__("json").dumps(session.to_dict()), encoding="utf-8")
    loaded = OfflineSessionAdapter().from_json(path)
    assert loaded.session_id == session.session_id
    assert signature(loaded.events[0]) == signature(device)
    assert loaded.provenance.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_wide_csv_with_units_and_program(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(
        "sequence,measured_at,received_at,phase,hr,spo2,program_id,intensity,duration_s\n"
        "3,2026-11-07T10:00:00+00:00,2026-11-07T10:00:01+00:00,during,70,0.96,M02,3,900\n",
        encoding="utf-8",
    )
    session = OfflineSessionAdapter().from_csv(
        path,
        session_id="s1",
        subject_ref="p1",
        device_ref="d1",
        units={"hr": "bpm", "spo2": "fraction"},
    )
    event = session.events[0]
    assert event.protocol.program_id == "M02"
    assert event.protocol.intensity == 3
    assert {m.metric: m.value for m in event.measurements} == {
        "heart_rate": 70.0,
        "spo2": 96.0,
    }


def test_stale_device_event_is_flagged():
    adapter = DeviceEventAdapter(stale_after_s=10)
    payload = {
        "session_id": "s1",
        "subject_ref": "p1",
        "device_ref": "d1",
        "measured_at": ts(0).isoformat(),
        "received_at": ts(60).isoformat(),
        "sequence": 0,
        "measurements": {"hr": 70},
    }
    event = adapter.parse_event(payload)
    assert event.quality == "stale"
    assert not event.is_valid


def test_missing_timestamp_is_rejected():
    with pytest.raises(SchemaError, match="timestamp"):
        DeviceEventAdapter().parse_event(
            {"session_id": "s1", "subject_ref": "p1", "device_ref": "d1"}
        )
