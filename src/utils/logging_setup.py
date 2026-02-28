"""
Unified logging setup for capturing all output to file.

This captures:
- Python print() statements
- Python logging
- Spark JVM logs
"""
import sys
import logging
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """Redirect stdout/stderr to both console and file."""

    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.terminal = sys.stdout if 'stdout' in str(file_path) else sys.stderr

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def setup_logging(log_file=None, log_dir='logs'):
    """
    Setup comprehensive logging to file.

    Args:
        log_file: Specific log file name (default: auto-generated with timestamp)
        log_dir: Directory for log files (default: 'logs')

    Returns:
        Path to log file
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"run_{timestamp}.log"

    log_file_path = log_path / log_file

    print(f"=" * 80)
    print(f"Logging to: {log_file_path}")
    print(f"=" * 80)

    # Redirect stdout and stderr to both console and file
    sys.stdout = TeeOutput(log_file_path, 'w')
    sys.stderr = TeeOutput(log_file_path, 'a')

    # Setup Python logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.__stdout__)  # Original stdout
        ]
    )

    print(f"✓ Logging initialized at {datetime.now()}")
    print(f"✓ All output will be written to: {log_file_path}")
    print("=" * 80)

    return log_file_path


def configure_spark_logging(spark, log_level='ERROR'):
    """
    Configure Spark JVM logging.

    Args:
        spark: SparkSession
        log_level: Logging level (ERROR, WARN, INFO, DEBUG)
    """
    # Set Spark context log level
    spark.sparkContext.setLogLevel(log_level)

    # Get Spark's log4j logger
    log4j = spark._jvm.org.apache.log4j
    logger = log4j.LogManager.getRootLogger()
    logger.setLevel(log4j.Level.toLevel(log_level))

    print(f"✓ Spark logging configured: {log_level}")


def log_memory_usage():
    """Log current memory usage."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)

    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    available_gb = vm.available / (1024 ** 3)
    used_pct = vm.percent

    print(f"💾 Memory Usage:")
    print(f"   Process: {mem_gb:.2f} GB")
    print(f"   System: {used_pct:.1f}% ({total_gb - available_gb:.1f} / {total_gb:.1f} GB)")
    print(f"   Available: {available_gb:.1f} GB")


def log_stage_start(stage_name):
    """Log the start of a pipeline stage."""
    print("\n" + "=" * 80)
    print(f"🚀 STAGE: {stage_name}")
    print(f"   Started at: {datetime.now()}")
    print("=" * 80)
    log_memory_usage()


def log_stage_complete(stage_name):
    """Log the completion of a pipeline stage."""
    print("\n" + "-" * 80)
    print(f"✅ COMPLETED: {stage_name}")
    print(f"   Finished at: {datetime.now()}")
    print("-" * 80)
    log_memory_usage()
