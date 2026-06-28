from __future__ import annotations

from .pipeline import _parse_args, _resolve_experiment_dir, run_vlm_stage


def main() -> None:
    args = _parse_args()
    result = run_vlm_stage(args)
    exp_dir = _resolve_experiment_dir(args)
    print(f"VLM views: {result['vlm_view_indices']}")
    print(f"Saved VLM outputs: {exp_dir / 'rgb_vlm_out'}")


if __name__ == "__main__":
    main()
