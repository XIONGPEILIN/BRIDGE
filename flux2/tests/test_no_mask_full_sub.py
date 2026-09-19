import unittest

import torch

from subject_driven_generation_pipeline import build_sparse_sub_branch


class NoMaskFullSubTest(unittest.TestCase):
    def test_full_image_mask_builds_main_sized_sub_for_both_variants(self):
        token_h, token_w = 4, 5
        main_latents = torch.zeros((1, 32, token_h, token_w), dtype=torch.float32)
        full_image_mask = torch.ones((token_h * 16, token_w * 16), dtype=torch.float32)

        variants = (
            {"sub_region_mode": "mask", "use_sparse_sub_branch": True, "pe_exchange_region": "mask"},
            {"sub_region_mode": "bbox", "use_sparse_sub_branch": False, "pe_exchange_region": "bbox"},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                selection, sub_latents, sub_ids = build_sparse_sub_branch(
                    main_latents,
                    full_image_mask,
                    sub_t_coord=60,
                    generator=torch.Generator().manual_seed(0),
                    **variant,
                )

                expected_tokens = token_h * token_w
                self.assertEqual(selection.crop_bounds, (0, token_h, 0, token_w))
                self.assertEqual(sub_latents.shape[1], expected_tokens)
                self.assertEqual(sub_ids.shape[1], expected_tokens)
                torch.testing.assert_close(
                    selection.main_token_indices,
                    torch.arange(expected_tokens),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    selection.ref_token_indices,
                    torch.arange(expected_tokens),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
