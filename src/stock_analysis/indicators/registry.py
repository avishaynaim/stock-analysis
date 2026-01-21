"""
Indicator registry for managing indicator definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class IndicatorDefinition:
    """Definition of a technical indicator."""

    name: str
    group: str
    func: Callable
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    required_periods: int = 50
    dependencies: list[str] = field(default_factory=list)
    requires_benchmark: bool = False


class IndicatorRegistry:
    """
    Registry for indicator definitions.

    Allows registration and lookup of indicators by name or group.
    """

    _indicators: dict[str, IndicatorDefinition] = {}
    _groups: dict[str, list[str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        group: str,
        func: Callable,
        description: str = "",
        parameters: Optional[dict[str, Any]] = None,
        required_periods: int = 50,
        dependencies: Optional[list[str]] = None,
        requires_benchmark: bool = False,
    ) -> None:
        """Register an indicator."""
        definition = IndicatorDefinition(
            name=name,
            group=group,
            func=func,
            description=description,
            parameters=parameters or {},
            required_periods=required_periods,
            dependencies=dependencies or [],
            requires_benchmark=requires_benchmark,
        )

        cls._indicators[name] = definition

        if group not in cls._groups:
            cls._groups[group] = []
        if name not in cls._groups[group]:
            cls._groups[group].append(name)

    @classmethod
    def get(cls, name: str) -> Optional[IndicatorDefinition]:
        """Get indicator by name."""
        return cls._indicators.get(name)

    @classmethod
    def get_group(cls, group: str) -> list[IndicatorDefinition]:
        """Get all indicators in a group."""
        names = cls._groups.get(group, [])
        return [cls._indicators[name] for name in names if name in cls._indicators]

    @classmethod
    def get_all_groups(cls) -> list[str]:
        """Get all group names."""
        return list(cls._groups.keys())

    @classmethod
    def get_all(cls) -> dict[str, IndicatorDefinition]:
        """Get all registered indicators."""
        return cls._indicators.copy()

    @classmethod
    def get_by_group(cls, group: str) -> dict[str, IndicatorDefinition]:
        """Get all indicators in a group as a dictionary."""
        names = cls._groups.get(group, [])
        return {name: cls._indicators[name] for name in names if name in cls._indicators}

    @classmethod
    def get_instance(cls) -> "IndicatorRegistry":
        """Get singleton instance of registry."""
        return cls


def indicator(
    name: str,
    group: str,
    description: str = "",
    parameters: Optional[dict[str, Any]] = None,
    required_periods: int = 50,
    dependencies: Optional[list[str]] = None,
    requires_benchmark: bool = False,
) -> Callable:
    """
    Decorator to register an indicator function.

    Usage:
        @indicator("rsi", "momentum", "Relative Strength Index")
        def compute_rsi(prices, period=14):
            ...
    """

    def decorator(func: Callable) -> Callable:
        IndicatorRegistry.register(
            name=name,
            group=group,
            func=func,
            description=description,
            parameters=parameters,
            required_periods=required_periods,
            dependencies=dependencies,
            requires_benchmark=requires_benchmark,
        )
        return func

    return decorator
