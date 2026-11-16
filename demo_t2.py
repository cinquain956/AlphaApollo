"""One-command entry point for Task 2.

Examples:
    python demo_t2.py dry-run
    python demo_t2.py run
    python demo_t2.py analyze
"""

from alphaapollo.experiments.t2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
