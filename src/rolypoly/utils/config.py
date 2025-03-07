"""Configuration management utilities for RolyPoly.

This module provides classes and functions for managing RolyPoly configuration,
including handling parameters like input/output paths, logging settings,
and resource allocation.

Example:
    ```python
    config = BaseConfig(output="output_dir", threads=4, memory="8gb")
    print(config.memory["giga"])
    # Output: 8
    ```
"""

from typing import Dict, Any, Optional 
from pathlib import Path
import json
import logging
from rolypoly.utils.loggit import setup_logging
from rich.console import Console
import re
import os
from datetime import datetime
console = Console()

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
        tmp_dir (str, optional): Temporary directory path. 
        overwrite (bool, optional): Whether to overwrite existing files.
        datadir (str, optional): Data directory path. 
        log_level (str, optional): Logging level ("debug", "info", "warning", "error", "critical"). 
        keep_tmp (bool, optional): Whether to keep temporary files.

    Example:
        ```python
        config = BaseConfig(
            output="results",
            threads=4,
            memory="8gb",
            log_level="debug"
        )
        print(config.memory["giga"])
        # Output: 8
        ```
    """
    def __init__(self,
                 output: Optional[str] = "rp_out", 
                 config_file: Optional[Path] = None, 
                 threads: int = 1, 
                 memory = '6gb', 
                 log_file: Optional[Path] = None, 
                 input: Optional[str] = None, 
                 tmp_dir: Optional[str] = None, 
                 overwrite: bool = False,
                 datadir: Optional[str] = None,
                 log_level: str = "info",
                 keep_tmp: bool = False):
        self.input = input
        self.threads = threads
        # print(memory)
        self.memory = self.parse_memory(memory)
        self.config_file = config_file
        self.log_file = log_file
        self.log_level = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}.get(log_level, logging.INFO)
        self.logger = self.setup_logger()
        self.tmp_dir = tmp_dir
        self.overwrite = overwrite
        # self.logger.info(f"Output directory: {output_dir}")
        # self.logger.info(f"output: {output}")
        self.output = Path(output)
        self.output_dir = self.output if self.output.is_dir() else self.output
        # self.logger.info(f"Output directory: {self.output_dir}")
        # self.logger.info(f"output: {self.output}")
        self.datadir = datadir or os.environ.get("ROLYPOLY_DATA")
        self.keep_tmp = keep_tmp
        
        if not overwrite:
            if not self.output_dir.exists():
                # self.logger.warning(f"Output directory does not exist: {self.output_dir}")
                self.logger.warning(f"Creating output directory: {self.output_dir}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.logger.info(f"Output directory {self.output_dir} already exists")
                if any(self.output_dir.iterdir()):
                    self.logger.warning("Output directory is not empty and overwrite is set to False. This might cause unexpected behavior.")
                
        if tmp_dir is not None:
            # check if tmp_dir exists
            if Path(tmp_dir).exists():
                self.tmp_dir = Path(tmp_dir).absolute().resolve()
                self.logger.debug(f"Temporary directory {self.tmp_dir} already exists, using it as is")
                if overwrite:
                    import shutil
                    self.logger.debug(f"overwrite set to True, Overwriting temporary directory {self.tmp_dir}")
                    shutil.rmtree(self.tmp_dir)
                    self.tmp_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.tmp_dir = Path(tmp_dir)
                self.logger.debug(f"Temporary directory {self.tmp_dir} does not exist, creating it")
                self.tmp_dir.mkdir(parents=True, exist_ok=True)

        else:
            self.tmp_dir = self.output_dir / f"rolypoly_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.debug(f"Creating temporary directory: {self.tmp_dir}")
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
                
    
    def setup_logger(self) -> logging.Logger:
        if isinstance(self.log_file, logging.Logger):
            return self.log_file
        return setup_logging(self.log_file, log_level=self.log_level)

    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items() if k != 'logger'}

    @classmethod
    def read(cls, config_file: Path):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, output_path: Path):
        with open(output_path, 'w') as f:
            tmp_dict = self.to_dict()
            for key, value in tmp_dict.items():
                if not isinstance(value, (str, int, float, bool, type(None))): 
                    tmp_dict[key] = str(value)
            json.dump(tmp_dict, f, indent=4)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def parse_memory(self, memory_str: str) -> Dict[str, str]:
        """Parse memory string (e.g., '6gb', '6000mb', '6g') into a dictionary with bytes, mega, giga."""
        import re
        
        # Convert memory string to lowercase and remove spaces
        memory_str = memory_str.lower().replace(" ", "")
        
        # Extract number and unit using regex
        match = re.match(r"(\d+)([kmgt]?b?)", memory_str)
        if not match:
            raise ValueError(f"Invalid memory format: {memory_str}. Expected format: NUMBER[UNIT] (e.g., 6gb, 6000mb, 6g)")
        
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
            "tb": 1024 * 1024 * 1024 * 1024
        }
        
        unit = unit.lower()
        if unit not in multipliers:
            raise ValueError(f"Invalid memory unit: {unit}")
        
        bytes_value = number * multipliers[unit]
        
        return {
            "bytes": f"{bytes_value}b",
            "mega": f"{bytes_value // (1024 * 1024)}m",
            "giga": f"{bytes_value // (1024 * 1024 * 1024)}g",
            "tera": f"{bytes_value // (1024 * 1024 * 1024 * 1024)}t"
        }

    def __str__(self):
        return f"BaseConfig(output={self.output}, output_dir={self.output_dir})"
