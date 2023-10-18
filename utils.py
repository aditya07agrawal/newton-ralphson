"""Utilities"""

from __future__ import annotations

from typing import ClassVar

from attrs import define, field


@define
class CountMixin:
    """Mixin to add counting functionality"""

    count: ClassVar[int] = 0

    index: int = field(init=False)

    def _set_index(self):
        self.index = type(self).count
        type(self).count += 1
