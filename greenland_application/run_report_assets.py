"""Refresh the report result snapshot and rebuild all report figures."""
import argparse

from .reporting import build_report_figures, refresh_report_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="Copy selected local checkpoints into the Git-trackable docs snapshot.",
    )
    args = parser.parse_args()

    if args.refresh_snapshot:
        refresh_report_snapshot()
    build_report_figures()


if __name__ == "__main__":
    main()
