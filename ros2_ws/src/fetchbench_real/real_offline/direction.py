from __future__ import annotations

from .pipeline import _parse_args, _resolve_experiment_dir, run_direction_stage


def main() -> None:
    args = _parse_args()
    result = run_direction_stage(args)
    exp_dir = _resolve_experiment_dir(args)
    print(f"Geometry-only direction: {result['geometry_only_direction']}")
    print(f"VLM-only direction: {result['vlm_only_direction']}")
    print(f"Aggregate direction alpha_vlm={float(args.alpha_vlm):.3f}: {result['aggregate_direction']}")
    print(f"Saved directions: {exp_dir / 'directions'}")


if __name__ == "__main__":
    main()
