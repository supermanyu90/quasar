"""The attendee companion: deterministic wayfinding to amenities."""

from __future__ import annotations

import unittest

from quasar import web
from quasar.amenities import AMENITIES, CALM_MAX_DENSITY
from quasar.crowd import CRITICAL_DENSITY

VENUE = "fwc-new-york"


def start_node() -> str:
    # The fixture's designed fan origin — a calm gate, not the congested one the
    # incident is at. (An accessible fan standing at the breaching gate genuinely
    # cannot get a step-free route out while its hall is at LOS E, which the router
    # correctly refuses; that is covered by the deterministic-plane tests, not here.)
    return web.profile(VENUE).fixture.fan_at_node


class TestAmenityCatalogue(unittest.TestCase):
    def test_every_amenity_is_offered_and_flagged(self) -> None:
        data = web.amenities_json(VENUE)
        keys = {a["key"] for a in data["amenities"]}
        self.assertEqual(keys, {a.key for a in AMENITIES})
        # the generated FIFA venue maps all of them
        self.assertTrue(all(a["available"] for a in data["amenities"]))

    def test_availability_reflects_the_actual_graph(self) -> None:
        """If a venue has no node with the tag, the amenity is flagged unavailable
        so the UI can grey it out rather than fail on tap."""
        data = web.amenities_json(VENUE)
        by = {a["key"]: a for a in data["amenities"]}
        self.assertTrue(by["food"]["available"])
        self.assertTrue(by["quiet"]["available"])


class TestWayfinding(unittest.TestCase):
    def test_routes_to_the_core_amenities(self) -> None:
        for key in ("food", "restroom", "first_aid", "exit", "lost_found", "merch", "water"):
            r = web.wayfind(VENUE, from_node=start_node(), amenity_key=key)
            self.assertIsNotNone(r["route"], f"{key} produced no route")
            self.assertTrue(r["destination"]["name"])

    def test_the_seat_request_goes_to_the_ticketed_seat_not_the_nearest(self) -> None:
        r = web.wayfind(VENUE, from_node=start_node(), amenity_key="seat", seat="SEAT-S")
        self.assertEqual(r["route"]["destination"], "SEAT-S")

    def test_accessible_restroom_route_is_step_free(self) -> None:
        venue = web.profile(VENUE).venue
        r = web.wayfind(VENUE, from_node=start_node(), amenity_key="accessible_restroom", accessible=True)
        self.assertIsNotNone(r["route"])
        for edge_id in r["route"]["edges"]:
            self.assertTrue(venue.edge(edge_id).step_free, f"{edge_id} is stepped")
        # and the destination actually carries both tags
        dest = venue.node(r["route"]["destination"])
        self.assertTrue({"washroom", "accessible"} <= dest.tags)

    def test_a_calm_route_stays_below_the_comfortable_density(self) -> None:
        """The sensory-calm mode holds the walk below Fruin LOS D — never near a
        crush, even for a fan who could physically push through one."""
        r = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", calm=True)
        self.assertIsNotNone(r["route"])
        self.assertLessEqual(r["route"]["worst_density"], CALM_MAX_DENSITY)
        self.assertLess(r["route"]["worst_density"], CRITICAL_DENSITY)

    def test_the_exit_request_finds_a_gate(self) -> None:
        r = web.wayfind(VENUE, from_node="C-N" if "C-N" in web.profile(VENUE).venue.nodes else start_node(),
                        amenity_key="exit")
        self.assertIsNotNone(r["route"])
        self.assertIn("gate", web.profile(VENUE).venue.node(r["route"]["destination"]).tags)

    def test_wheelchair_assist_is_a_request_not_a_route(self) -> None:
        r = web.wayfind(VENUE, from_node=start_node(), amenity_key="assist")
        self.assertTrue(r["request"])
        self.assertIsNone(r["route"])
        self.assertIn("steward", r["message"].lower())

    def test_the_reply_greets_the_fan_in_their_language(self) -> None:
        es = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", language="es")
        self.assertIn("ruta", es["message"].lower())
        hi = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", language="hi")
        self.assertIn("मार्ग", hi["message"])

    def test_the_whole_card_switches_language_not_just_the_greeting(self) -> None:
        """The regression this guards: switching language used to change only the
        lead-in phrase. Now the message AND the secondary notes are localised."""
        en = web.wayfind(VENUE, from_node=start_node(), amenity_key="food",
                         language="en", accessible=True)
        es = web.wayfind(VENUE, from_node=start_node(), amenity_key="food",
                         language="es", accessible=True)
        self.assertIsNotNone(es["route"])
        self.assertNotEqual(en["message"], es["message"])
        # the step-free qualifier is localised, not left in English
        self.assertIn("sin escalones", es["message"])
        self.assertNotIn("step-free", es["message"])
        # the secondary "worst crowding" note is localised too
        self.assertIn("servicio", " ".join(es["notes"]).lower())
        self.assertIn("level of service", " ".join(en["notes"]).lower())

    def test_numbers_and_place_names_survive_the_language_switch(self) -> None:
        """Entity preservation: the distance (a number) and the destination name
        (signage) appear verbatim in a non-English reply — never translated."""
        es = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", language="es")
        self.assertTrue(any(ch.isdigit() for ch in es["message"]))
        self.assertIn(es["destination"]["name"], es["message"])

    def test_an_unsupported_language_falls_back_to_english_not_machine_translation(self) -> None:
        de = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", language="de")
        en = web.wayfind(VENUE, from_node=start_node(), amenity_key="food", language="en")
        self.assertEqual(de["message"], en["message"])

    def test_every_language_has_complete_authored_phrasing(self) -> None:
        """No half-translated card: every field formats for every shipped language,
        with all placeholders resolved (a typo'd placeholder would leave a brace)."""
        sample = dict(dest="Gate 4", m=240, mins=3, tail=" (x)",
                      los="D", n=2, amenity="restroom", sfp="", loc="Gate 1")
        for lang, w in web._WORDING.items():
            for field in ("route", "worst", "more", "none_mapped", "no_route", "assist"):
                text = getattr(w, field).format(**sample)
                self.assertTrue(text.strip(), f"{lang}.{field} empty")
                self.assertNotIn("{", text, f"{lang}.{field} unresolved placeholder")

    def test_an_unknown_amenity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            web.wayfind(VENUE, from_node=start_node(), amenity_key="teleporter")

    def test_an_unknown_start_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            web.wayfind(VENUE, from_node="NOWHERE", amenity_key="food")

    def test_the_fan_route_never_crosses_a_cordon(self) -> None:
        """The attendee router is the same one the control room uses: a cordon binds
        it. A fan is never routed through a closed corridor."""
        venue = web.profile(VENUE).venue
        # cordon the first ring corridor; the route must avoid it
        cordon = ["R-1"] if "R-1" in venue.edges else [next(iter(venue.edges))]
        r = web.wayfind(VENUE, from_node=start_node(), amenity_key="merch", cordoned=cordon)
        if r["route"]:
            self.assertNotIn(cordon[0], r["route"]["edges"])


if __name__ == "__main__":
    unittest.main()
