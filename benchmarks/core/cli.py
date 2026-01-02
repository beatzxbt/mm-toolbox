"""CLI argument parsing utilities.

Provides builder pattern for composable command-line interface construction
with common benchmark arguments.
"""

from __future__ import annotations

import argparse


class BenchmarkCLI:
    """Builder for benchmark command-line interfaces.

    Provides consistent CLI construction with common benchmark arguments
    pre-configured. Uses builder pattern for extensibility.

    Args:
        description: Benchmark description for --help.
    """

    def __init__(self, description: str) -> None:
        """Initialize CLI builder.

        Args:
            description: Benchmark description for --help.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_args()

    def _add_common_args(self) -> None:
        """Add common arguments used by most benchmarks."""
        self.parser.add_argument(
            "--operations",
            "-n",
            type=int,
            default=100_000,
            help="Number of operations to benchmark (default: 100,000)",
        )
        self.parser.add_argument(
            "--warmup",
            "-w",
            type=int,
            default=1_000,
            help="Number of warmup operations (default: 1,000)",
        )
        self.parser.add_argument(
            "--multi-size",
            "-m",
            action="store_true",
            help="Test multiple sizes",
        )

    def add_size_arg(
        self, default: int = 1024, help_text: str | None = None
    ) -> BenchmarkCLI:
        """Add --size/-s argument.

        Args:
            default: Default size value.
            help_text: Custom help text (defaults to mentioning the default).

        Returns:
            Self for method chaining.
        """
        self.parser.add_argument(
            "--size",
            "-s",
            type=int,
            default=default,
            help=help_text or f"Size parameter (default: {default})",
        )
        return self

    def add_comparison_flag(self, name: str, help_text: str) -> BenchmarkCLI:
        """Add a comparison flag (e.g., --compare-async).

        Args:
            name: Flag name (e.g., "async" for --compare-async).
            help_text: Help text for the flag.

        Returns:
            Self for method chaining.
        """
        self.parser.add_argument(
            f"--compare-{name}",
            action="store_true",
            help=help_text,
        )
        return self

    def add_input_file(
        self, default: str, help_text: str | None = None
    ) -> BenchmarkCLI:
        """Add --input/-i argument for data files.

        Args:
            default: Default input file path.
            help_text: Custom help text (defaults to mentioning the default).

        Returns:
            Self for method chaining.
        """
        self.parser.add_argument(
            "--input",
            "-i",
            default=default,
            help=help_text or f"Input file path (default: {default})",
        )
        return self

    def parse(self) -> argparse.Namespace:
        """Parse command-line arguments.

        Returns:
            Parsed arguments namespace.
        """
        return self.parser.parse_args()
