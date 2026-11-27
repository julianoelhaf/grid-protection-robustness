import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="grid-protection-robustness")
    parser.add_argument("study", choices=("eusipco", "dpsp"))
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--sample-dir", type=Path, help="Pinned episode pickle files")
    smoke.add_argument("--jobs", type=int, default=1)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("window_ms", type=int, choices=(10, 20, 30, 40, 50))
    case = sub.add_parser("case")
    case.add_argument("case", type=int, choices=range(260), metavar="0-259")
    case.add_argument("--jobs", type=int, default=1)
    remaining = sub.add_parser("remaining")
    remaining.add_argument("--array", action="store_true")
    remaining.add_argument("--active", default="", help="Comma-separated queued or running case IDs")
    summarize = sub.add_parser("summarize")
    verify = sub.add_parser("verify")
    verify.add_argument("--supplement", type=Path, required=True)
    for command in (smoke, prepare, case, remaining, summarize):
        command.add_argument("--run", type=Path, required=True)
    for command in (smoke, prepare):
        command.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if args.study == "dpsp":
        parser.error("DPSP is pending provenance and redistribution clearance.")
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs must be positive")

    from . import execution
    from .audit import verify_archive
    from .jobs import claim
    from .run_matrix import smoke_checks
    from .summarize import summarize as aggregate

    if args.command == "smoke":
        with claim(args.run, "smoke", wait=True):
            execution.bind_run(args.run, "smoke")
            smoke_checks(args.data, args.run, args.sample_dir)
            for case_id in (44, 62, 68, 86):
                execution.run_case(args.run, case_id, args.jobs, smoke=True)
    elif args.command == "prepare":
        execution.prepare_case_inputs(args.run, args.data, args.window_ms)
    elif args.command == "case":
        execution.run_case(args.run, args.case, args.jobs)
    elif args.command == "remaining":
        active = {int(value) for value in args.active.split(",") if value}
        execution.remaining(args.run, active, args.array)
    elif args.command == "summarize":
        with claim(args.run, "summarize", wait=True):
            for case_id in execution.matrix().case:
                if execution.read_receipt(args.run, case_id) is None:
                    parser.error(f"Missing case {case_id}")
            aggregate(args.run, args.run)
    else:
        verify_archive(args.supplement)
