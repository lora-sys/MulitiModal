"""仓库脚手架 smoke（#51）。

只证明测试基础设施可用；真正的契约 / 模型测试由 #25 / #48 / #49 / #42 落位。
"""

import sys


def test_python_environment_is_usable():
    assert sys.version_info >= (3, 11)
