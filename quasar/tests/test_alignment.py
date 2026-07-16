"""The guide is the problem statement mapped to the console. These tests keep it
honest: every track and every persona named in the challenge must be represented,
and every 'show me' deep link must point at a place the console can actually reach.
If a feature is renamed or a tab removed, one of these fails before a judge finds a
dead link."""

from __future__ import annotations

import unittest

from quasar import alignment, web

# The four tracks and four personas the challenge names, by title fragment.
REQUIRED_TRACKS = ("crowd", "navigation", "decision", "language")
REQUIRED_PERSONAS = ("fan", "organizer", "volunteer", "staff")

# Query keys the console's boot() knows how to honour (app.js deep links).
KNOWN_PARAMS = {"venue", "tab", "view", "find", "mode", "run", "approve", "theme"}
KNOWN_TABS = {"guide", "control", "fan", "ready", "premat", "audit"}


class TestGuideCoverage(unittest.TestCase):
    def test_serialised_guide_has_every_section(self) -> None:
        guide = web.guide_json()
        self.assertTrue(guide["challenge"])
        self.assertTrue(guide["thesis"])
        keys = {s["key"] for s in guide["sections"]}
        self.assertEqual(keys, {"tracks", "personas", "objectives"})

    def test_all_four_tracks_present(self) -> None:
        keys = {t.key for t in alignment.TRACKS}
        self.assertEqual(keys, set(REQUIRED_TRACKS))

    def test_all_four_personas_present(self) -> None:
        keys = {p.key for p in alignment.PERSONAS}
        self.assertEqual(keys, set(REQUIRED_PERSONAS))

    def test_every_item_is_complete(self) -> None:
        for _key, _heading, items in alignment.sections():
            for it in items:
                for field in (it.icon, it.title, it.summary, it.how, it.cta):
                    self.assertTrue(field.strip(), f"{it.key}: empty field")
                self.assertTrue(it.where, f"{it.key}: no deep link")


class TestDeepLinksAreReachable(unittest.TestCase):
    def test_every_where_uses_known_params(self) -> None:
        for _key, _heading, items in alignment.sections():
            for it in items:
                self.assertLessEqual(
                    set(it.where), KNOWN_PARAMS,
                    f"{it.key}: unknown deep-link param",
                )

    def test_every_tab_target_exists(self) -> None:
        for _key, _heading, items in alignment.sections():
            for it in items:
                tab = it.where.get("tab")
                if tab is not None:
                    self.assertIn(tab, KNOWN_TABS, f"{it.key}: tab '{tab}'")

    def test_every_venue_target_exists(self) -> None:
        known = {v["id"] for v in web.venues_json()["venues"]}
        for _key, _heading, items in alignment.sections():
            for it in items:
                venue = it.where.get("venue")
                if venue is not None:
                    self.assertIn(venue, known, f"{it.key}: venue '{venue}'")

    def test_every_find_target_is_a_real_amenity(self) -> None:
        from quasar.amenities import BY_KEY
        for _key, _heading, items in alignment.sections():
            for it in items:
                find = it.where.get("find")
                if find is not None:
                    self.assertIn(find, BY_KEY, f"{it.key}: amenity '{find}'")


if __name__ == "__main__":
    unittest.main()
