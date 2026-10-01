"""采样时刻补点规划核心逻辑（与 Web 层解耦，便于测试）。

目标：针对当地 22:00–次日 06:00 的夜间窗口，在已采样时刻之外补入
**数量最少**的新增采样时刻，使得：

1. 窗口首尾均被覆盖（存在采样时刻 ≤ 窗口起点、≥ 窗口终点）；
2. 所有与窗口有正重叠的相邻采样间隔不超过 15 分钟；
3. 补点只允许落在窗口内；窗口外仅以紧邻边界的已有时刻作为锚点。

存在多个等优（同为最少点数）方案时，取按绝对时刻排序后字典序最早的
方案，即“优先从较早的可行时刻补起”。

全部计算以 UTC 绝对时刻进行，天然正确处理夏令时春跳/秋拨。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .assessment import MAX_GAP, AssessmentError, Window, elapsed, night_window

_UTC = timezone.utc


@dataclass(frozen=True)
class SamplingPlan:
    """补点规划结果（补点时刻已换算到施工地时区）。"""

    window: Window
    added_times: list[datetime]
    already_compliant: bool

    @property
    def added_count(self) -> int:
        return len(self.added_times)


def plan_sampling(
    tz: ZoneInfo,
    night_date: date,
    sampled_times: list[datetime] | None,
) -> SamplingPlan:
    """生成最少数量的新增采样时刻。

    sampled_times 必须为带时区偏移的 aware datetime，内部只按绝对时刻比较；
    可使用任意偏移提交。重复或乱序（按绝对时刻）抛 AssessmentError。
    """
    window = night_window(tz, night_date)
    times = _validate_and_utc(list(sampled_times or []))
    added_utc, already_compliant = _plan_for_window(
        times, window.start.astimezone(_UTC), window.end.astimezone(_UTC)
    )
    return SamplingPlan(
        window=window,
        added_times=[t.astimezone(tz) for t in added_utc],
        already_compliant=already_compliant,
    )


def _validate_and_utc(sampled_times: list[datetime]) -> list[datetime]:
    """统一换算到 UTC，并拒绝缺偏移、重复或乱序的输入。"""
    times: list[datetime] = []
    for t in sampled_times:
        if t.tzinfo is None or t.utcoffset() is None:
            raise AssessmentError(
                "naive_timestamp", f"采样时间缺少时区偏移：{t.isoformat()}"
            )
        times.append(t.astimezone(_UTC))

    # 拒绝重复时间与乱序（输入整体校验，相邻比较即可覆盖）
    for prev, cur in zip(times, times[1:]):
        if cur == prev:
            raise AssessmentError("duplicate_timestamp", f"采样时间重复：{cur.isoformat()}")
        if cur < prev:
            raise AssessmentError(
                "out_of_order",
                f"采样时间乱序：{cur.isoformat()} 早于前一时刻 {prev.isoformat()}",
            )
    return times


def _plan_for_window(
    times: list[datetime], w_start: datetime, w_end: datetime
) -> tuple[list[datetime], bool]:
    """绝对时刻窗口上的通用补点算法，返回 (UTC 补点列表, 原数据是否已合格)。"""
    # 参与规划的锚点：窗口内全部时刻 + 窗口外紧邻边界的各一个时刻
    pre_window = [t for t in times if t < w_start]
    in_window = [t for t in times if w_start <= t <= w_end]
    post_window = [t for t in times if t > w_end]
    anchors: list[datetime] = sorted(
        (pre_window[-1:] if pre_window else [])
        + in_window
        + (post_window[:1] if post_window else [])
    )

    already_compliant = _is_compliant(anchors, w_start, w_end)

    added: list[datetime] = []
    # 补齐未覆盖的窗口首尾（强制补在边界上）
    if not anchors or anchors[0] > w_start:
        added.append(w_start)
    if not anchors or anchors[-1] < w_end:
        added.append(w_end)

    merged = sorted(set(anchors + added))

    # 逐段处理与窗口有正重叠、且间隔超过 15 分钟的相邻锚点对
    fixed = set(merged)
    for a, b in zip(merged, merged[1:]):
        clip_start = max(a, w_start)
        clip_end = min(b, w_end)
        if clip_end <= clip_start:
            continue  # 与窗口无正重叠
        if elapsed(a, b) <= MAX_GAP:
            continue  # 恰好 15 分钟允许

        # 从右端每 15 分钟反推布点；反推点越出窗口时截到边界，并以边界点为
        # 新的反推基准继续（窗外到边界的边与窗口零重叠、不受 15 分钟限制）。
        # 逐点加入直到整段满足约束为止；这种右对齐方案同时是字典序最早的
        # 最少补点方案（每个补点都取其右邻约束允许的最早时刻）。
        fills: list[datetime] = []
        cursor = b
        max_steps = int(elapsed(a, b) / MAX_GAP) + 3  # 防御性上界
        steps = 0
        while not _pair_covered(a, b, fills, w_start, w_end):
            steps += 1
            if steps > max_steps:  # pragma: no cover - 理论不可达
                raise RuntimeError("补点规划未能收敛")
            t = cursor - MAX_GAP
            if t > w_end:
                t = w_end
            elif t < w_start:
                t = w_start
            cursor = t  # 截到边界后，后续反推从边界点重新对齐
            if a < t < b and t not in fixed and t not in fills:
                fills.append(t)
        added.extend(fills)

    return sorted(set(added)), already_compliant


def _pair_covered(
    a: datetime,
    b: datetime,
    fills: list[datetime],
    w_start: datetime,
    w_end: datetime,
) -> bool:
    """锚点对 (a, b) 连同已补点后，所有与窗口正重叠的相邻间隔是否均 ≤ 15 分钟。"""
    seq = [a] + sorted(fills) + [b]
    for x, y in zip(seq, seq[1:]):
        clip_start = max(x, w_start)
        clip_end = min(y, w_end)
        if clip_end > clip_start and elapsed(x, y) > MAX_GAP:
            return False
    return True


def _is_compliant(
    anchors: list[datetime], w_start: datetime, w_end: datetime
) -> bool:
    """原数据是否已满足采样充分性（首尾覆盖 + 重叠间隔均 ≤ 15 分钟）。"""
    if not anchors or anchors[0] > w_start or anchors[-1] < w_end:
        return False
    for a, b in zip(anchors, anchors[1:]):
        clip_start = max(a, w_start)
        clip_end = min(b, w_end)
        if clip_end <= clip_start:
            continue
        if elapsed(a, b) > MAX_GAP:
            return False
    return True
