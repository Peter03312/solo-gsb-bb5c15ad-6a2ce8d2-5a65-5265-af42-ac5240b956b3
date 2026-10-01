"""采样补点规划核心逻辑测试。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.assessment import MAX_GAP, AssessmentError, elapsed, night_window
from app.sampling import plan_sampling

CST = ZoneInfo("Asia/Shanghai")
NY = ZoneInfo("America/New_York")
D = date(2026, 9, 25)

# at(0) = 北京 22:00（窗口起点，UTC 14:00）；at(8) = 次日北京 06:00
_BASE = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)


def at(h: float, tz=timezone.utc) -> datetime:
    return (_BASE + timedelta(hours=h)).astimezone(tz)


def hm(t: datetime) -> str:
    return t.strftime("%H:%M")


def add_hours(plan):
    return [
        (t.astimezone(timezone.utc) - at(0)).total_seconds() / 3600
        for t in plan.added_times
    ]


# ---------- 空输入与首尾覆盖 ----------

def test_empty_sampling_fills_whole_8h_window():
    p = plan_sampling(CST, D, [])
    assert p.already_compliant is False
    assert p.added_count == 33  # 22:00, 22:15, ..., 06:00
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[-1]) == "06:00"
    # 全部补点严格递增、间隔恰好 15 分钟、带 +08:00 偏移
    for prev, cur in zip(p.added_times, p.added_times[1:]):
        assert elapsed(prev, cur) == MAX_GAP
        assert cur.utcoffset() == timedelta(hours=8)


def test_empty_sampling_list_means_none_taken():
    # 核心层空列表等价于“尚无采样”（null/缺省在 API schema 层归一为空列表）
    p = plan_sampling(CST, D, [])
    assert p.added_count == 33
    assert p.already_compliant is False


def test_inner_only_sample_requires_both_boundaries():
    # 窗口内只有 02:00 一个点：首尾都要补，段内再按 15 分钟补
    p = plan_sampling(CST, D, [at(4)])
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[-1]) == "06:00"
    assert p.added_count == 32
    assert p.already_compliant is False


def test_boundary_samples_avoid_extra_boundary_points():
    # 恰好 22:00 / 06:00 已采样：不重复补边界
    p = plan_sampling(CST, D, [at(0), at(8)])
    assert hm(p.added_times[0]) == "22:15"
    assert hm(p.added_times[-1]) == "05:45"
    assert p.added_count == 31


# ---------- 合格判定 ----------

def test_already_compliant_returns_empty():
    # 窗口两端各外延 5 分钟、窗内每 10 分钟一个采样
    times = [at(-1 / 12) + timedelta(minutes=10 * i) for i in range(50)]
    p = plan_sampling(CST, D, times)
    assert p.already_compliant is True
    assert p.added_count == 0
    assert p.added_times == []


def test_exact_15_min_grid_compliant():
    times = [at(-0.25 + 0.25 * i) for i in range(35)]  # 21:45 .. 06:15
    p = plan_sampling(CST, D, times)
    assert p.already_compliant is True
    assert p.added_count == 0


def test_outside_large_gap_ignored():
    # 20:00 -> 22:00 间隔 2 小时，但不与窗口重叠，不影响合格性
    times = [at(-2), at(0)] + [at(10 / 60 * i) for i in range(1, 49)]
    p = plan_sampling(CST, D, times)
    assert p.already_compliant is True
    assert p.added_count == 0


# ---------- 最少点数与字典序最早 ----------

def test_minimal_count_20m_gap():
    # 22:00、22:20、06:00：20 分钟空档只需 1 个补点
    p = plan_sampling(CST, D, [at(0), at(1 / 3), at(8)])
    assert p.added_count == 31  # 1 + 30（22:20 之后到 06:00 整段）
    first = add_hours(p)[0]
    assert first == pytest.approx(5 / 60, abs=1e-9)  # 22:05 而非 22:15


def test_minimal_count_40m_gap_two_fills():
    # 22:00、22:40、06:00：40 分钟空档最少 2 个补点
    p = plan_sampling(CST, D, [at(0), at(2 / 3), at(8)])
    assert add_hours(p)[:2] == pytest.approx([10 / 60, 25 / 60], abs=1e-9)


def test_16_min_gap_single_fill_right_aligned():
    # 22:00、22:16：1 个补点应落在 22:01（右对齐），让左段 1 分钟、右段 15 分钟
    p = plan_sampling(CST, D, [at(0), at(16 / 60), at(8)])
    assert add_hours(p)[0] == pytest.approx(1 / 60, abs=1e-9)


def test_result_satisfies_all_constraints():
    """随机感的混合锚点：补点后整体必须全部满足约束。"""
    times = [at(-0.5), at(0.75), at(2.5), at(7.2), at(8.4)]
    p = plan_sampling(CST, D, times)
    w = night_window(CST, D)
    existing_utc = [t.astimezone(timezone.utc) for t in times]
    seq = sorted(existing_utc + [t.astimezone(timezone.utc) for t in p.added_times])
    # 首尾覆盖
    assert seq[0] <= w.start.astimezone(timezone.utc)
    assert seq[-1] >= w.end.astimezone(timezone.utc)
    # 正重叠间隔 ≤ 15 分钟
    for a, b in zip(seq, seq[1:]):
        cs = max(a, w.start.astimezone(timezone.utc))
        ce = min(b, w.end.astimezone(timezone.utc))
        if ce > cs:
            assert elapsed(a, b) <= MAX_GAP
    # 补点只落在窗口内
    for t in p.added_times:
        u = t.astimezone(timezone.utc)
        assert w.start.astimezone(timezone.utc) <= u <= w.end.astimezone(timezone.utc)


# ---------- 窗口外锚点 ----------

def test_outside_anchor_within_15_min_no_boundary_cut():
    # 21:50 -> 22:20：30 分钟间隔，1 个补点 22:05 即可（窗外段不重叠）
    p = plan_sampling(CST, D, [at(-10 / 60), at(20 / 60), at(8)])
    assert add_hours(p)[0] == pytest.approx(5 / 60, abs=1e-9)
    assert hm(p.added_times[0]) == "22:05"


def test_outside_anchor_far_requires_boundary_cut():
    # 21:40 -> 22:20：40 分钟，右对齐补点 22:05 距 21:40 仍有 25 分钟，
    # 必须在 22:00 再切一点（窗外段不重叠，故 21:40->22:00 不受约束）
    p = plan_sampling(CST, D, [at(-20 / 60), at(20 / 60), at(8)])
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[1]) == "22:05"


def test_far_outside_anchor_right_side_cut():
    # 05:50 -> 08:00：06:00 后到 08:00 不重叠，06:00 必须补点
    p = plan_sampling(CST, D, [at(0), at(8 - 10 / 60), at(10)])
    assert hm(p.added_times[-1]) == "06:00"


def test_only_far_outside_anchors_both_sides():
    # 20:00 与次日 08:00：窗口两侧都无近锚点，边界都补点，窗内按 15 分钟补齐
    p = plan_sampling(CST, D, [at(-2), at(10)])
    assert p.added_count == 33
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[-1]) == "06:00"


def test_only_nearest_outside_anchor_used():
    # 窗口前有多个点，只有最接近 22:00 的 21:59 参与：
    # 21:59 -> 22:30 与窗口在 22:00–22:30 正重叠，31 分钟超限；
    # 右对齐反推先补 22:15，但 (21:59, 22:15) 仍正重叠 16 分钟，
    # 故还需在 22:00 切点（若误用更早点 21:50，结果相同——窗外部分不受约束）。
    times = [at(-3), at(-2), at(-1 / 60), at(0.5), at(9), at(11)]
    p = plan_sampling(CST, D, times)
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[1]) == "22:15"
    assert hm(p.added_times[-1]) == "06:00"
    assert p.already_compliant is False


# ---------- 错误输入 ----------

def test_duplicate_timestamp_rejected():
    with pytest.raises(AssessmentError) as exc:
        plan_sampling(CST, D, [at(1), at(1)])
    assert exc.value.code == "duplicate_timestamp"


def test_out_of_order_rejected():
    with pytest.raises(AssessmentError) as exc:
        plan_sampling(CST, D, [at(2), at(1)])
    assert exc.value.code == "out_of_order"


def test_naive_timestamp_rejected():
    with pytest.raises(AssessmentError, match="时区"):
        plan_sampling(CST, D, [datetime(2026, 9, 25, 22, 0)])


def test_invalid_timezone_rejected():
    from app.assessment import get_timezone

    with pytest.raises(AssessmentError, match="时区"):
        plan_sampling(get_timezone("Mars/Olympus"), D, [])


# ---------- 偏移与夏令时 ----------

def test_input_offsets_accepted_output_in_local_tz():
    tokyo = timezone(timedelta(hours=9))
    times = [
        datetime(2026, 9, 25, 23, 0, tzinfo=tokyo),   # 北京 22:00
        datetime(2026, 9, 26, 7, 0, tzinfo=tokyo),    # 北京 06:00
    ]
    p = plan_sampling(CST, D, times)
    assert all(t.utcoffset() == timedelta(hours=8) for t in p.added_times)
    assert hm(p.added_times[0]) == "22:15"
    assert p.added_count == 31


def test_dst_spring_forward_real_seconds_grid():
    # 美东 2026-03-07 晚窗口为 7 小时；空输入按真实秒数每 15 分钟补点
    p = plan_sampling(NY, date(2026, 3, 7), [])
    assert elapsed(p.window.start, p.window.end) == timedelta(hours=7)
    assert p.added_count == 29  # 28 个 15 分钟间隔
    for prev, cur in zip(p.added_times, p.added_times[1:]):
        assert elapsed(prev, cur) == MAX_GAP
    # 首点 -05:00（EST），末点 -04:00（EDT）
    assert p.added_times[0].utcoffset() == timedelta(hours=-5)
    assert p.added_times[-1].utcoffset() == timedelta(hours=-4)
    assert hm(p.added_times[0]) == "22:00"
    assert hm(p.added_times[-1]) == "06:00"


def test_dst_fall_back_real_seconds_grid():
    # 美东 2026-10-31 晚窗口为 9 小时
    p = plan_sampling(NY, date(2026, 10, 31), [])
    assert elapsed(p.window.start, p.window.end) == timedelta(hours=9)
    assert p.added_count == 37  # 36 个 15 分钟间隔
    for prev, cur in zip(p.added_times, p.added_times[1:]):
        assert elapsed(prev, cur) == MAX_GAP
    assert p.added_times[0].utcoffset() == timedelta(hours=-4)
    assert p.added_times[-1].utcoffset() == timedelta(hours=-5)


def test_dst_spring_gap_spanning_skip_real_seconds():
    # 春跳：22:00 与一个绝对时刻 60 分钟后的点，跨跳变挂钟只显示 59 分钟差，
    # 但按真实秒数间隔 60 分钟 > 15，必须补点
    w = night_window(NY, date(2026, 3, 7))
    later = w.start + timedelta(minutes=60)
    p = plan_sampling(NY, date(2026, 3, 7), [w.start, later, w.end])
    assert p.added_count >= 3
    assert hm(p.added_times[0]) != hm(later.astimezone(NY))
