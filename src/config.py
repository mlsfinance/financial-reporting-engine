"""
configuration management for the financial_reporting_engine project

this module provides a small wrapper around a YAML configuration file, offering:
    * loading and validation of the YAML config
    * access to nested keys using dot notation (for example "data.input_path")
    * convenience methods to retrieve typed values

the idea is to keep configuration outside the code so the same pipeline can be reused for different datasets and reporting setups by just changing the YAML.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml


class ConfigError(Exception):
    """this is raised when there's a configuration-related error"""


T = TypeVar("T")


@dataclass
class Config:
    path: Path
    data: Dict[str, Any]


    #constructors
    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """to load configuration from a YAML file"""
        path = Path(path)


        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")


        try:
            with path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Error parsing YAML config '{path}': {exc}") from exc


        if not isinstance(raw_data, dict):
            raise ConfigError(
                f"Top-level structure in config must be a mapping, got {type(raw_data)}"
            )


        return cls(path=path, data=raw_data)




    #core access methods
    def _resolve_key(self, key: str) -> Any:
        parts = key.split(".")
        current: Any = self.data


        for part in parts:
            if not isinstance(current, dict):
                raise ConfigError(
                    f"Cannot descend into '{key}': '{part}' is not a mapping "
                    f"(current type: {type(current)})"
                )
            if part not in current:
                raise ConfigError(f"Missing config key: '{key}' (stopped at '{part}')")
            current = current[part]


        return current


    def get(self, key: str, default: Optional[T] = None, type_: Optional[Type[T]] = None) -> Optional[T]:
        try:
            value = self._resolve_key(key)
        except ConfigError:
            return default


        if type_ is not None and value is not None:
            if not isinstance(value, type_):
                #it tries a simple cast if possible (for example: "123" -> int)
                try:
                    value = type_(value)  #type: ignore[call-arg]
                except Exception as exc:  #noqa: BLE001
                    raise ConfigError(
                        f"Config key '{key}' expected type {type_.__name__}, "
                        f"got {type(value).__name__}"
                    ) from exc



        return value  #type: ignore[return-value]


    def require(self, key: str, type_: Optional[Type[T]] = None) -> T:
        try:
            value = self._resolve_key(key)
        except ConfigError as exc:
            raise ConfigError(f"Required config key missing: '{key}'") from exc

        if type_ is not None and value is not None:
            if not isinstance(value, type_):
                #it tries a simple cast if possible
                try:
                    value = type_(value)  #type: ignore[call-arg]
                except Exception as exc:  #noqa: BLE001
                    raise ConfigError(
                        f"Config key '{key}' expected type {type_.__name__}, "
                        f"got {type(value).__name__}"
                    ) from exc


        return value  #type: ignore[return-value]

    #this function sets a config value using dotted notation, for example: cfg.set("report.output_pdf", "./reports/x.pdf")
    def set(self, key: str, value) -> None:
        parts = key.split(".")
        d = self.data

        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]

        d[parts[-1]] = value


    #helper methods for common sections
    def data_section(self) -> Dict[str, Any]:
        """return the 'data' section as a dict (or empty dict if missing)"""
        value = self.get("data", default={}, type_=dict)
        return value or {}


    def report_section(self) -> Dict[str, Any]:
        """return the 'report' section as a dict (or empty dict if missing)"""
        value = self.get("report", default={}, type_=dict)
        return value or {}


if __name__ == "__main__":
    #basic manual test when running this module directly
    example_path = Path(__file__).parent.parent / "config" / "example.yaml"
    try:
        cfg = Config.from_file(example_path)
        print(f"Loaded config from: {cfg.path}")
        print("data.input_path =", cfg.get("data.input_path"))
        print("report.output_pdf =", cfg.get("report.output_pdf"))
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")