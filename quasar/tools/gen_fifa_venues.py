"""Generate the 16 FIFA World Cup 2026 venue specs from public metadata.

    PYTHONPATH=src python3 tools/gen_fifa_venues.py

Writes venues/fwc-*.json. Each is a *representative* model fitted to the venue's
real capacity and gate count -- correct for scale and planning, explicitly not a
surveyed floor plan (every spec is stamped topology: "representative", and the
readiness audit and console say so). Re-run to regenerate; output is deterministic
in the venue id.
"""

from __future__ import annotations

import json
from pathlib import Path

from quasar.venue_factory import FWC_2026_VENUES, generate_spec
from quasar.venue_spec import load_spec

VENUES = Path(__file__).resolve().parent.parent / "venues"


def main() -> None:
    for meta in FWC_2026_VENUES:
        spec = generate_spec(meta)
        # Load it before writing: a spec that would not load is a bug in the factory,
        # and it must fail here, not on match night.
        profile = load_spec(spec)
        path = VENUES / f"{meta['id']}.json"
        path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
        print(
            f"  {meta['name']:<26} {meta['city']:<15} "
            f"{profile.capacity:>7,}  {len(profile.venue.nodes):>3}n {len(profile.venue.edges):>3}e  "
            f"{'/'.join(profile.languages)}"
        )
    print(f"\nwrote {len(FWC_2026_VENUES)} venue specs to {VENUES}")


if __name__ == "__main__":
    main()
