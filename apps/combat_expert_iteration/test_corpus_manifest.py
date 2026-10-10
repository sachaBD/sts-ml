import unittest
from .corpus_manifest import (LEGACY_ELIGIBLE, TEN_DECK_LOCATOR, TEN_DECK_SOURCE_RUN, collection_plan,
                              eligible_legacy_sources, parse_manifest, validate_production_freeze)


def manifest():
    families = [dict(family_id=f"old{i}", role="original", wave=None, provenance=[f"old-{i}"]) for i in range(10)]
    families += [dict(family_id=f"w{w}-{i}", role="pilot", wave=w, provenance=[f"new-{w}-{i}"]) for w in range(1, 5) for i in range(10)]
    families += [dict(family_id="dev", role="development", wave=None, provenance=["dev-id"]), dict(family_id="final", role="final", wave=None, provenance=["final-id"])]
    return dict(schema="champ-corpus-pilot-v1", legacy_run=dict(run_id=TEN_DECK_SOURCE_RUN, locator="unfrozen"), families=families,
                historical_exposure=[dict(source_id="bootstrap", family_id="old0", category="bootstrap", used_for_training=True),
                                     dict(source_id="update007", family_id="old0", category="learner_update007", used_for_training=True)],
                replay_source_ids=["bootstrap"])


class ManifestTests(unittest.TestCase):
    def test_replay_eligibility_is_separate_from_historical_exposure(self):
        parsed = parse_manifest(manifest())
        self.assertEqual([x.source_id for x in eligible_legacy_sources(parsed)], ["bootstrap"])
        self.assertEqual(len(parsed.sha256), 64); self.assertIn("learner_update006", LEGACY_ELIGIBLE)
        bad = manifest(); bad["replay_source_ids"] = ["update007"]
        with self.assertRaisesRegex(ValueError, "ineligible"): parse_manifest(bad)
        bad = manifest(); bad["legacy_run"] = dict(run_id="some-other-run", locator="x")
        with self.assertRaisesRegex(ValueError, "ten-deck"): parse_manifest(bad)

    def test_provenance_and_freeze_requirements_fail_closed(self):
        bad = manifest(); bad["families"][-1]["provenance"] = ["old-0"]
        with self.assertRaisesRegex(ValueError, "belongs to both"): parse_manifest(bad)
        with self.assertRaisesRegex(ValueError, "20 development"): validate_production_freeze(parse_manifest(manifest()))
        full = manifest()
        full["families"] += [dict(family_id=f"dev{i}", role="development", wave=None, provenance=[f"d{i}"]) for i in range(19)]
        full["families"] += [dict(family_id=f"final{i}", role="final", wave=None, provenance=[f"f{i}"]) for i in range(99)]
        full["legacy_run"].update(locator=TEN_DECK_LOCATOR, artifact_sha256={"model.pt": "0" * 64})
        validate_production_freeze(parse_manifest(full))

    def test_collection_originals_never_appear_and_all_round_quotas(self):
        parsed = parse_manifest(manifest())
        for round_number, total, values in ((1, 1000, set()), (2, 1500, {50}), (3, 1500, {25}), (4, 1500, {16, 17})):
            jobs = collection_plan(parsed, round_number, 9)
            revisits = [x for x in jobs if x["source"] == "revisit_learner"]
            self.assertEqual(sum(x["games"] for x in jobs), total)
            self.assertEqual({x["games"] for x in revisits}, values)
            self.assertFalse(any(x["family_id"].startswith("old") for x in jobs))
        with self.assertRaises(ValueError): collection_plan(parsed, True)
        bad = manifest(); bad["families"][10]["wave"] = True
        with self.assertRaises(ValueError): parse_manifest(bad)


if __name__ == "__main__": unittest.main()
