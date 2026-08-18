import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rolypoly.utils.logging.loggit import setup_logging

DEFAULT_MEMORY = "8g"


class BaseConfig:
    """Configuration manager for RolyPoly commands.

    A dictionary-like object that holds parameters common to many RolyPoly commands.
    Handles configuration of input/output paths, logging, resource allocation,
    and temporary file management.

    Args:
        output (str, optional): Output directory or file path.
        config_file (Path, optional): Path to a JSON configuration file.
        threads (int, optional): Number of CPU threads to use.
        memory (str, optional): Memory allocation (e.g., "8g", "8000m").
        log_file (Path, optional): Path to log file.
        input (str, optional): Input file or directory path.
        temp_dir (str, optional): Temporary directory path. if not provided, a temporary directory will be created in the output directory.
        overwrite (bool, optional): Whether to overwrite existing files.
        datadir (str, optional): Data directory path.
        log_level (str, optional): Logging level ("debug", "info", "warning", "error", "critical").
        keep_tmp (bool, optional): Whether to keep temporary files.
    """

    def __init__(
        self,
        input: Union[Path, str],
        output: Union[Path, str],
        config_file: Union[Path, str, None] = None,
        threads: Optional[int] = 1,
        memory: Optional[str] = None,
        log_file: Union[Path, logging.Logger, None] = None,
        temp_dir: Optional[str] = None,
        overwrite: bool = False,
        datadir: Optional[str] = None,
        log_level: str = "INFO",
        keep_tmp: bool = False,
    ):
        import shutil
        from datetime import datetime

        # Basic parameter initialization
        self.input = input
        self.threads = threads
        self.memory = self.parse_memory(memory or DEFAULT_MEMORY)
        self.config_file = Path(config_file) if config_file else None
        self.log_file = log_file or Path.cwd() / "rolypoly.log"
        self.overwrite = overwrite
        self.output = Path(output)
        self.output_dir = self.output if self.output.is_dir() else self.output
        self.datadir = datadir or os.environ.get("ROLYPOLY_DATA")
        self.keep_tmp = keep_tmp

        # Set up logging
        self.log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(log_level.lower(), logging.INFO)
        self.logger = self.setup_logger()

        # Create output directory if needed
        if not overwrite and not self.output_dir.exists():
            self.logger.info(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        elif not overwrite and any(self.output_dir.iterdir()):
            self.logger.warning(
                "Output directory is not empty and overwrite is set to False. This might cause unexpected behavior."
            )

        # define temporary directory
        self.temp_dir = (
            Path(temp_dir).absolute()
            if temp_dir
            else (
                self.output_dir
                / f"rolypoly_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ).absolute()
        )

        # clean existing temporary directory if overwrite is set to True
        if self.temp_dir.exists():
            if overwrite:
                self.logger.debug(
                    f"Cleaning existing temporary directory {self.temp_dir}"
                )
                shutil.rmtree(self.temp_dir)
            else:
                self.logger.warning(
                    f"Temporary directory {self.temp_dir} already exists. Set overwrite to True to clean it."
                )

        # create temporary directory
        self.logger.debug(f"Creating temporary directory: {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def setup_logger(self) -> logging.Logger:
        """Setup logger for the configuration"""
        if isinstance(self.log_file, logging.Logger):
            return self.log_file
        return setup_logging(self.log_file, log_level=self.log_level)

    def cleanup_temp(self) -> None:
        """Remove this config's temporary directory unless ``keep_tmp`` is set.

        ``__init__`` always creates ``self.temp_dir`` (an auto-named
        ``rolypoly_tmp_<timestamp>`` under the output dir when no explicit
        ``temp_dir`` was given). Commands MUST call this at the end so that
        auto-created temp dirs are not leaked into the output directory. Safe to
        call multiple times. Never removes the output directory itself.
        """
        try:
            if getattr(self, "keep_tmp", False):
                return
            temp_dir = getattr(self, "temp_dir", None)
            if not temp_dir:
                return
            temp_dir = Path(temp_dir)
            # Guard against accidentally removing the output dir (e.g. if a caller
            # passed temp_dir == output_dir).
            if temp_dir.resolve() == self.output_dir.resolve():
                return
            if temp_dir.exists():
                import shutil
                self.logger.debug(f"Removing temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:  # never let cleanup break a completed run
            self.logger.warning(f"Could not clean temp dir: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if k != "logger"
        }

    @classmethod
    def read(cls, config_file: Path):
        """Read configuration from JSON file"""
        import json

        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, output_path: Path):
        """Save configuration to JSON file"""
        import json

        with open(output_path, "w") as f:
            tmp_dict = self.to_dict()
            for key, value in tmp_dict.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    tmp_dict[key] = str(value)
            json.dump(tmp_dict, f, indent=4)

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse_memory(self, memory_str: str) -> Dict[str, str]:
        """Parse memory strings such as ``8g`` or ``8000m`` into unit variants."""
        import re

        # Convert memory string to lowercase and remove spaces
        memory_str = memory_str.lower().replace(" ", "")

        # Extract number and unit using regex
        match = re.match(r"(\d+)([kmgt]?b?)", memory_str)
        if not match:
            raise ValueError(
                f"Invalid memory format: {memory_str}. Expected format: NUMBER[UNIT] (e.g., 8g or 8000m)"
            )

        number, unit = match.groups()
        number = int(number)

        # Convert to bytes based on unit
        multipliers = {
            "": 1,  # no unit assumes bytes
            "k": 1024,
            "m": 1024 * 1024,
            "g": 1024 * 1024 * 1024,
            "t": 1024 * 1024 * 1024 * 1024,
            "kb": 1024,
            "mb": 1024 * 1024,
            "gb": 1024 * 1024 * 1024,
            "tb": 1024 * 1024 * 1024 * 1024,
        }

        unit = unit.lower()
        if unit not in multipliers:
            raise ValueError(f"Invalid memory unit: {unit}")

        bytes_value = number * multipliers[unit]

        return {
            "bytes": f"{bytes_value}b",
            "mega": f"{bytes_value // (1024 * 1024)}m",
            "giga": f"{bytes_value // (1024 * 1024 * 1024)}g",
            "tera": f"{bytes_value // (1024 * 1024 * 1024 * 1024)}t",
        }

    def __str__(self):
        return f"BaseConfig(output={self.output}, output_dir={self.output_dir})"
