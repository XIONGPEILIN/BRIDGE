import unittest

import torch

from subject_driven_generation_pipeline import build_sparse_sub_branch


class ReleaseContractTest(unittest.TestCase):
    def build(self, main, mask, *, sparse=True):
        return build_sparse_sub_branch(
            main, mask, sub_t_coord=20,
            sub_region_mode="mask" if sparse else "bbox",
            use_sparse_sub_branch=sparse,
            pe_exchange_region="mask" if sparse else "bbox",
            generator=torch.Generator(device="cpu").manual_seed(17),
        )

    def test_sub_noise_is_independent_of_main_values(self):
        mask = torch.ones(64, 80)
        _, sub_a, ids_a = self.build(torch.zeros(1, 32, 4, 5), mask)
        _, sub_b, ids_b = self.build(torch.full((1, 32, 4, 5), 1234.0), mask)
        torch.testing.assert_close(sub_a, sub_b, rtol=0, atol=0)
        torch.testing.assert_close(ids_a, ids_b, rtol=0, atol=0)
        self.assertTrue(torch.all(ids_a[..., 0] == 20))
        self.assertGreater(float(sub_a.abs().sum()), 0)

    def test_interior_hole_removes_only_one_sparse_token(self):
        main = torch.zeros(1, 32, 4, 5)
        full = torch.ones(64, 80)
        hole = full.clone()
        hole[16:32, 16:32] = 0
        all_sel, all_sub, _ = self.build(main, full)
        sel, sub, ids = self.build(main, hole)
        self.assertEqual(sel.crop_bounds, all_sel.crop_bounds)
        self.assertEqual(sub.shape[1], 19)
        expected = torch.tensor([i for i in range(20) if i != 6])
        torch.testing.assert_close(sel.main_token_indices, expected)
        torch.testing.assert_close(sel.ref_token_indices, torch.arange(19))
        torch.testing.assert_close(sub, all_sub[:, expected], rtol=0, atol=0)
        self.assertEqual(ids.shape[1], 19)

    def test_dense_bbox_keeps_interior_hole(self):
        main = torch.zeros(1, 32, 4, 5)
        full = torch.ones(64, 80)
        hole = full.clone()
        hole[16:32, 16:32] = 0
        _, all_sub, _ = self.build(main, full, sparse=False)
        sel, sub, _ = self.build(main, hole, sparse=False)
        self.assertEqual(sub.shape[1], 20)
        torch.testing.assert_close(sel.main_token_indices, torch.arange(20))
        torch.testing.assert_close(sub, all_sub, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
