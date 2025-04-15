import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from rolypoly.utils.loggit import setup_logging


class BaseConfig:
    """Configuration manager for RolyPoly commands.

    A dictionary-like object that holds parameters common to many RolyPoly commands.
    Handles configuration of input/output paths, logging, resource allocation,
    and temporary file management.

    Args:
        output (str, optional): Output directory or file path.
        config_file (Path, optional): Path to a JSON configuration file.
        threads (int, optional): Number of CPU threads to use.
        memory (str, optional): Memory allocation (e.g., "6gb", "8000mb").
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
        output: Optional[str] = "rp_out",
        config_file: Optional[Path] = None,
        threads: Optional[int] = 1,
        memory: Optional[str] = "6gb",
        log_file: Optional[Path] = None,
        input: Optional[str] = None,
        temp_dir: Optional[str] = None,
        overwrite: bool = False,
        datadir: Optional[str] = None,
        log_level: str = "info",
        keep_tmp: bool = False,
    ):
        import shutil
        from datetime import datetime

        # Basic parameter initialization
        self.input = input
        self.threads = threads
        self.memory = self.parse_memory(memory)
        self.config_file = Path(config_file) if config_file else None
        self.log_file = Path(log_file) if log_file else None
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
        }.get(log_level, logging.INFO)
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
        """Parse memory string (e.g., '6gb', '6000mb', '6g') into a dictionary with bytes, mega, giga"""
        import re

        # Convert memory string to lowercase and remove spaces
        memory_str = memory_str.lower().replace(" ", "")

        # Extract number and unit using regex
        match = re.match(r"(\d+)([kmgt]?b?)", memory_str)
        if not match:
            raise ValueError(
                f"Invalid memory format: {memory_str}. Expected format: NUMBER[UNIT] (e.g., 6gb, 6000mb, 6g)"
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
