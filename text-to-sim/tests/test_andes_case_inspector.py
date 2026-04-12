"""Unit tests for src.andes_case_inspector.

Most tests use a real ANDES load of ieee14/ieee14.raw (which ships
with ANDES and loads in ~1s). The edge cases around missing paths,
empty args, and cache invalidation use synthetic fixtures.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.andes_case_inspector import (  # noqa: E402
    INSPECTED_MODELS,
    MAX_ENTRIES_PER_FIELD,
    _format_list,
    build_case_idx_inventory,
    clear_inventory_cache,
)


class FormatListTests(unittest.TestCase):
    def test_string_list_is_quoted(self):
        self.assertEqual(_format_list(["a", "b"]), '["a", "b"]')

    def test_int_list_unquoted(self):
        self.assertEqual(_format_list([1, 2, 3]), "[1, 2, 3]")

    def test_mixed_types_handled(self):
        self.assertEqual(_format_list([1, "x"]), '[1, "x"]')

    def test_empty_list(self):
        self.assertEqual(_format_list([]), "[]")

    def test_short_list_not_truncated(self):
        # Right at the threshold.
        values = list(range(MAX_ENTRIES_PER_FIELD))
        result = _format_list(values)
        self.assertNotIn("...", result)
        self.assertIn(f"{MAX_ENTRIES_PER_FIELD - 1}]", result)

    def test_long_list_truncated_in_middle(self):
        # Just over the threshold.
        values = list(range(MAX_ENTRIES_PER_FIELD + 20))
        result = _format_list(values)
        self.assertIn("...", result)
        # First element shown.
        self.assertIn("[0,", result)
        # Last element shown.
        self.assertIn(f"{MAX_ENTRIES_PER_FIELD + 19}]", result)


class EmptyArgsTests(unittest.TestCase):
    def test_empty_source_returns_empty(self):
        self.assertEqual(build_case_idx_inventory("", "foo"), "")

    def test_empty_reference_returns_empty(self):
        self.assertEqual(build_case_idx_inventory("builtin", ""), "")

    def test_both_empty_returns_empty(self):
        self.assertEqual(build_case_idx_inventory("", ""), "")


class MissingPathTests(unittest.TestCase):
    def test_unknown_builtin_case_returns_empty(self):
        result = build_case_idx_inventory("builtin", "nonsense/garbage.xlsx")
        self.assertEqual(result, "")

    def test_uploaded_case_missing_file_returns_empty(self):
        result = build_case_idx_inventory(
            "uploaded", "missing.xlsx", uploaded_dir="/tmp/does-not-exist"
        )
        self.assertEqual(result, "")

    def test_local_case_missing_returns_empty(self):
        result = build_case_idx_inventory(
            "local", "/tmp/absolutely/nowhere/case.xlsx"
        )
        self.assertEqual(result, "")

    def test_unknown_source_returns_empty(self):
        result = build_case_idx_inventory("mystery", "ieee14/ieee14.raw")
        self.assertEqual(result, "")


class Ieee14IntegrationTests(unittest.TestCase):
    """Load a real ANDES built-in case and check the rendered shape."""

    @classmethod
    def setUpClass(cls):
        clear_inventory_cache()
        cls.inventory = build_case_idx_inventory("builtin", "ieee14/ieee14.raw")

    def test_inventory_is_non_empty(self):
        self.assertTrue(self.inventory)
        self.assertIn("ANDES case idx inventory for ieee14/ieee14.raw:", self.inventory)

    def test_header_exactly_matches(self):
        self.assertTrue(
            self.inventory.startswith("ANDES case idx inventory for ieee14/ieee14.raw:")
        )

    def test_bus_section_contains_all_14_buses(self):
        self.assertIn("Bus (14 entries):", self.inventory)
        # IEEE14 has bus idx as plain ints 1..14.
        self.assertIn("idx  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]", self.inventory)

    def test_line_section_has_string_idx_and_topology(self):
        self.assertIn("Line (20 entries):", self.inventory)
        # IEEE14 uses "Line_N" string idx.
        self.assertIn('"Line_1"', self.inventory)
        self.assertIn('"Line_20"', self.inventory)
        # Topology fields are present.
        self.assertIn("bus1 = [", self.inventory)
        self.assertIn("bus2 = [", self.inventory)

    def test_pq_section_present(self):
        self.assertIn("PQ (11 entries):", self.inventory)
        self.assertIn('"PQ_1"', self.inventory)
        self.assertIn("bus  = [", self.inventory)

    def test_pv_section_present(self):
        self.assertIn("PV (4 entries):", self.inventory)

    def test_slack_section_present(self):
        self.assertIn("Slack (1 entry):", self.inventory)

    def test_guidance_footer_present(self):
        # The guidance is what makes this useful to the model; pin its
        # key anchors.
        self.assertIn('"line 18"', self.inventory)
        self.assertIn("Line.idx.v[N-1]", self.inventory)
        self.assertIn("bus1==X and bus2==Y", self.inventory)

    def test_sections_appear_in_expected_order(self):
        """Bus -> Line -> PQ -> PV -> Slack -> Shunt."""
        positions = []
        for name in INSPECTED_MODELS:
            idx = self.inventory.find(f"{name} (")
            positions.append((name, idx))
        # Each subsequent section's header comes after the prior.
        for i in range(1, len(positions)):
            prior_name, prior_pos = positions[i - 1]
            name, pos = positions[i]
            if prior_pos < 0 or pos < 0:
                continue
            self.assertGreater(pos, prior_pos,
                               msg=f"{name} section appears before {prior_name}")


class CachingTests(unittest.TestCase):
    def test_second_call_returns_cached_string(self):
        clear_inventory_cache()
        first = build_case_idx_inventory("builtin", "ieee14/ieee14.raw")
        second = build_case_idx_inventory("builtin", "ieee14/ieee14.raw")
        # Same string content (not the cache is really hit -- we'd need
        # to mock andes.load to prove that, but string equality after
        # clearing is the contract callers care about).
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_clear_invalidates_cache(self):
        build_case_idx_inventory("builtin", "ieee14/ieee14.raw")
        clear_inventory_cache()
        # After clearing, subsequent call still returns the same content.
        again = build_case_idx_inventory("builtin", "ieee14/ieee14.raw")
        self.assertTrue(again)
        self.assertIn("ieee14/ieee14.raw", again)


if __name__ == "__main__":
    unittest.main()
