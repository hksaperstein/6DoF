"""Load the arm's kinematic configuration from arm.yaml.

The DH table is data, not code: rows reference parameters by name so a
parameter can change without touching kinematics.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

# "placeholder" and "assumed" are deliberately excluded: they carry real
# values that are swept or revisited routinely, and warning on every load
# would train the reader to ignore the open/suspect signal that does matter.
NOISY_STATUSES = {"open", "suspect", "unknown"}


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    status: str
    rationale: str


@dataclass(frozen=True)
class DHRow:
    joint: int
    d: float
    a: float
    alpha: float  # radians


@dataclass(frozen=True)
class ArmConfig:
    parameters: dict[str, Parameter]
    rows: tuple[DHRow, ...]
    _raw_rows: tuple[dict, ...]

    def value(self, name: str) -> float:
        if name not in self.parameters:
            raise KeyError(f"unknown parameter {name!r}")
        return self.parameters[name].value

    def with_override(self, **overrides: float) -> "ArmConfig":
        params = dict(self.parameters)
        for name, value in overrides.items():
            if name not in params:
                raise KeyError(f"cannot override unknown parameter {name!r}")
            params[name] = replace(params[name], value=float(value))
        return ArmConfig(
            parameters=params,
            rows=_build_rows(self._raw_rows, params),
            _raw_rows=self._raw_rows,
        )


def _resolve(entry, params: dict[str, Parameter]) -> float:
    """A row field is either a literal number or a parameter name."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if entry not in params:
        raise KeyError(f"row references unknown parameter {entry!r}")
    return params[entry].value


def _build_rows(raw_rows, params: dict[str, Parameter]) -> tuple[DHRow, ...]:
    return tuple(
        DHRow(
            joint=int(r["joint"]),
            d=_resolve(r["d"], params),
            a=_resolve(r["a"], params),
            alpha=math.radians(float(r["alpha"])),
        )
        for r in raw_rows
    )


def load_config(path: Path) -> ArmConfig:
    data = yaml.safe_load(Path(path).read_text())

    units = data.get("units", {})
    if units.get("length") != "mm" or units.get("angle") != "deg":
        raise ValueError(
            f"unsupported units {units!r}; only length: mm and angle: deg are implemented"
        )
    convention = data["dh"].get("convention")
    if convention != "classical":
        raise ValueError(
            f"unsupported DH convention {convention!r}; only 'classical' is implemented"
        )

    params: dict[str, Parameter] = {}
    for name, spec in data["parameters"].items():
        value = spec.get("value")
        status = spec.get("status", "unknown")
        if value is None:
            warnings.warn(
                f"parameter {name!r} has no value (status {status!r}); "
                f"loading as 0.0",
                UserWarning,
                stacklevel=2,
            )
            value = 0.0
        elif status in NOISY_STATUSES:
            warnings.warn(
                f"parameter {name!r} has status {status!r}; "
                f"its value {value} is not trustworthy",
                UserWarning,
                stacklevel=2,
            )
        params[name] = Parameter(
            name=name,
            value=float(value),
            status=status,
            rationale=spec.get("rationale", ""),
        )

    raw_rows = tuple(data["dh"]["rows"])
    rows = _build_rows(raw_rows, params)
    if [r.joint for r in rows] != list(range(len(rows))):
        raise ValueError("dh.rows must be in ascending joint order starting at 0")
    return ArmConfig(parameters=params, rows=rows, _raw_rows=raw_rows)
