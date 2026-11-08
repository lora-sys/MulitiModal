"""实时生理疲劳与安全推荐工程层（#95–#100）。

本包只提供**版本化协议、适配器、窗口与门禁**；没有真实配对标签和独立验证前，
任何疲劳分数都必须表达为 ``NOT_AVAILABLE`` / ``ABSTAIN``，不得用合成 smoke
指标冒充实测效果。
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION"]

SCHEMA_VERSION = "physio-v1"
