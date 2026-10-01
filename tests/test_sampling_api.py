"""/sampling-plan 接口测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# at(0)=北京22:00, at(8)=北京06:00
_BASE = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)


def at(h: float) -> datetime:
    return _BASE + timedelta(hours=h)


def req(sampled_times=None, tz="Asia/Shanghai", night="2026-09-25"):
    body = {"timezone": tz, "night_date": night}
    if sampled_times is not None:
        body["sampled_times"] = sampled_times
    return body


def iso(times):
    return [t.isoformat() for t in times]


def test_health_still_ok():
    assert client.get("/health").status_code == 200


def test_empty_array_plans_full_window():
    r = client.post("/sampling-plan", json=req([]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added_count"] == 33
    assert body["already_compliant"] is False
    assert body["added_times"][0] == "2026-09-25T22:00:00+08:00"
    assert body["added_times"][-1] == "2026-09-26T06:00:00+08:00"
    assert body["window"]["duration_seconds"] == 28800.0
    # 相邻补点恰好 15 分钟
    ts = [datetime.fromisoformat(s) for s in body["added_times"]]
    for prev, cur in zip(ts, ts[1:]):
        assert (cur - prev) == timedelta(minutes=15)


def test_null_sampled_times_accepted():
    r = client.post("/sampling-plan", json=req(None))
    assert r.status_code == 200, r.text
    assert r.json()["added_count"] == 33


def test_field_missing_defaults_to_empty():
    # sampled_times 缺省等价于空
    r = client.post("/sampling-plan", json={"timezone": "Asia/Shanghai", "night_date": "2026-09-25"})
    assert r.status_code == 200, r.text
    assert r.json()["added_count"] == 33


def test_compliant_input_returns_zero_added():
    # 21:45 起每 15 分钟至 06:15
    times = iso(at(-0.25 + 0.25 * i) for i in range(35))
    r = client.post("/sampling-plan", json=req(times))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["already_compliant"] is True
    assert body["added_count"] == 0
    assert body["added_times"] == []


def test_lex_earliest_20m_gap():
    times = iso([at(0), at(1 / 3), at(8)])
    r = client.post("/sampling-plan", json=req(times))
    body = r.json()
    assert body["added_times"][0] == "2026-09-25T22:05:00+08:00"


def test_duplicate_timestamp_400_same_error_format():
    times = iso([at(1), at(1), at(2)])
    r = client.post("/sampling-plan", json=req(times))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "duplicate_timestamp"
    assert "message" in err


def test_out_of_order_400_same_error_format():
    times = iso([at(2), at(1)])
    r = client.post("/sampling-plan", json=req(times))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "out_of_order"


def test_naive_time_400():
    r = client.post(
        "/sampling-plan",
        json=req(["2026-09-25T22:00:00", "2026-09-26T06:00:00"]),
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_bad_timezone_400():
    r = client.post("/sampling-plan", json=req([], tz="Mars/Olympus"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_timezone"


def test_bad_date_400():
    r = client.post(
        "/sampling-plan", json={"timezone": "Asia/Shanghai", "night_date": "2026-09-xx"}
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_missing_required_fields_400():
    r = client.post("/sampling-plan", json={"timezone": "Asia/Shanghai"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_non_array_sampled_times_400():
    r = client.post(
        "/sampling-plan",
        json={"timezone": "Asia/Shanghai", "night_date": "2026-09-25", "sampled_times": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_other_offset_input_output_in_local_tz():
    # 用 +09:00 偏移提交北京 22:00 / 06:00
    r = client.post(
        "/sampling-plan",
        json=req(["2026-09-25T23:00:00+09:00", "2026-09-26T07:00:00+09:00"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added_count"] == 31
    assert body["added_times"][0] == "2026-09-25T22:15:00+08:00"
    assert body["added_times"][-1] == "2026-09-26T05:45:00+08:00"


def test_zulu_offset_accepted():
    # 北京 22:00 = UTC 14:00
    r = client.post(
        "/sampling-plan",
        json=req(["2026-09-25T14:00:00Z", "2026-09-25T22:00:00Z"]),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["already_compliant"] is False
    assert body["added_count"] == 31
    assert body["added_times"][0] == "2026-09-25T22:15:00+08:00"
    assert body["added_times"][-1] == "2026-09-26T05:45:00+08:00"


def test_dst_spring_forward_response():
    # 美东 2026-03-07 晚窗口 7 小时：补 29 点
    r = client.post(
        "/sampling-plan", json=req([], tz="America/New_York", night="2026-03-07")
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["window"]["duration_seconds"] == 25200.0
    assert body["added_count"] == 29
    assert body["added_times"][0] == "2026-03-07T22:00:00-05:00"
    assert body["added_times"][-1] == "2026-03-08T06:00:00-04:00"


def test_dst_fall_back_response():
    # 美东 2026-10-31 晚窗口 9 小时：补 37 点
    r = client.post(
        "/sampling-plan", json=req([], tz="America/New_York", night="2026-10-31")
    )
    body = r.json()
    assert body["window"]["duration_seconds"] == 32400.0
    assert body["added_count"] == 37
    assert body["added_times"][0] == "2026-10-31T22:00:00-04:00"
    assert body["added_times"][-1] == "2026-11-01T06:00:00-05:00"
