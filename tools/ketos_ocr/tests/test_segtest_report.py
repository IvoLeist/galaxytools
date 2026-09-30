"""Regression coverage for compact pixel-metric arrays in Kraken 7.1.1."""
import unittest

from ketos_segtest import corrected_source


REPORT_LOOP = """
rows = []
for idx in pixel_idxs:
    rows.append((idx, class_pixel_accuracy[idx], class_iu[idx]))
"""


class SegtestReportTest(unittest.TestCase):
    def test_region_after_baseline_uses_compact_metric_position(self):
        # Class 2 is a baseline and has no pixel metrics. Region 3 occupies
        # position 2, not position 3, in both arrays.
        namespace = dict(pixel_idxs=[0, 1, 3],
                         class_pixel_accuracy=[0.1, 0.2, 0.9],
                         class_iu=[0.3, 0.4, 0.8])
        exec(corrected_source(REPORT_LOOP), namespace)
        self.assertEqual(namespace['rows'], [(0, 0.1, 0.3), (1, 0.2, 0.4), (3, 0.9, 0.8)])

    def test_multiple_gaps_preserve_class_ids_and_metric_order(self):
        namespace = dict(pixel_idxs=[0, 1, 4, 7],
                         class_pixel_accuracy=[0.1, 0.2, 0.3, 0.4],
                         class_iu=[0.5, 0.6, 0.7, 0.8])
        exec(corrected_source(REPORT_LOOP), namespace)
        self.assertEqual(namespace['rows'][-2:], [(4, 0.3, 0.7), (7, 0.4, 0.8)])

    def test_upstream_changes_require_review(self):
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            corrected_source(REPORT_LOOP.replace('for idx in pixel_idxs:', 'for idx in other:'))


if __name__ == '__main__':
    unittest.main()
