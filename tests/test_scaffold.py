"""仓库脚手架 smoke（#51）。

只证明测试基础设施可用；真正的契约 / 模型测试由 #25 / #48 / #49 / #42 落位。

这条版本断言不是自我证明的摆设：仓库刻意没有 `[project]` 表，因此也没有
`requires-python`（pyproject.toml:1-2 说明构建元数据等 #48 包结构落位后再引入），
所以它是 README「Python 3.11+」与 ruff `target-version = "py311"` 所依赖的那个
下限唯一的机械守卫。删掉它并不会去掉一个同义反复，只会让这个下限无人守护。
"""

import sys


def test_python_environment_is_usable():
    assert sys.version_info >= (3, 11)
