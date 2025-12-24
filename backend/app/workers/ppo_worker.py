from __future__ import annotations

import argparse

from app.workers.ppo_worker_runner import run_ppo_worker


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_ppo_worker(args.stage_id)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-id", type=int, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    main()

