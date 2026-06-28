from __future__ import annotations

from .pipeline import _parse_args, _resolve_experiment_dir, run_prep


def main() -> None:
    args = _parse_args()
    result = run_prep(args)
    exp_dir = _resolve_experiment_dir(args)
    usable = sum(1 for item in result["prepared_vlm_inputs"] if item.get("usable"))
    print(f"Prepared {usable} VLM input images: {exp_dir / 'rgb_vlm_in'}")
    print(f"Saved subset manifest: {exp_dir / 'vlm_subset.json'}")
    print(f"Mask placeholder: {exp_dir / 'masks' / 'masks_manifest.json'}")


if __name__ == "__main__":
    main()
