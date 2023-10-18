"""Utilities"""

from __future__ import annotations

from typing import ClassVar

from attrs import define, field


@define
class CountMixin:
    """Mixin to add counting functionatlity"""

    count: ClassVar[int] = 0

    index: int = field(init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.index: int = type(self).count
        type(self).count += 1
