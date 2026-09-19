import unittest

import torch

from qwen_pe_exchange_sparse_model import STE as HardSTE
from qwen_pe_exchange_sparse_soft_model import STE as SoftSTE


class STEPaddingMaskTest(unittest.TestCase):
    @staticmethod
    def _make_ste(ste_cls, *, regularized: bool = False):
        torch.manual_seed(1234)
        model = ste_cls(
            dim_in=16,
            num_layers=1,
            sampler="vanilla_ste",
            entropy_weight=1.0 if regularized else 0.0,
            sparsity_target=0.3 if regularized else None,
            sparsity_weight=1.0 if regularized else 0.0,
            head_init="normal",
            head_init_std=0.1,
            return_aux=True,
            encoder_layers=0 if regularized else 1,
            encoder_num_heads=8,
        )
        return model

    @staticmethod
    def _inputs():
        torch.manual_seed(5678)
        valid = torch.randn(1, 4, 16)
        padded_a = torch.randn(1, 3, 16)
        padded_b = torch.randn(1, 3, 16) * 50.0
        x_a = torch.cat([valid, padded_a], dim=1)
        x_b = torch.cat([valid, padded_b], dim=1)
        key_padding_mask = torch.tensor(
            [[False, False, False, False, True, True, True]], dtype=torch.bool
        )
        return valid, x_a, x_b, key_padding_mask

    def test_valid_logits_are_isolated_from_padding_contents(self):
        _, x_a, x_b, key_padding_mask = self._inputs()
        for ste_cls in (HardSTE, SoftSTE):
            with self.subTest(ste=ste_cls.__module__):
                model = self._make_ste(ste_cls).eval()
                with torch.no_grad():
                    _, aux_a = model(x_a, layer_idx=0, src_key_padding_mask=key_padding_mask)
                    _, aux_b = model(x_b, layer_idx=0, src_key_padding_mask=key_padding_mask)
                    _, aux_unmasked_a = model(x_a, layer_idx=0)
                    _, aux_unmasked_b = model(x_b, layer_idx=0)

                torch.testing.assert_close(
                    aux_a["logits"][:, :4],
                    aux_b["logits"][:, :4],
                    rtol=0.0,
                    atol=1e-6,
                )
                unmasked_delta = (
                    aux_unmasked_a["logits"][:, :4] - aux_unmasked_b["logits"][:, :4]
                ).abs().max()
                self.assertGreater(float(unmasked_delta), 1e-5)

    def test_omitted_mask_matches_explicit_none(self):
        _, x_a, _, _ = self._inputs()
        for ste_cls in (HardSTE, SoftSTE):
            with self.subTest(ste=ste_cls.__module__):
                model = self._make_ste(ste_cls).eval()
                with torch.no_grad():
                    gate_implicit, aux_implicit = model(x_a, layer_idx=0)
                    gate_explicit, aux_explicit = model(
                        x_a, layer_idx=0, src_key_padding_mask=None
                    )
                torch.testing.assert_close(gate_implicit, gate_explicit, rtol=0.0, atol=0.0)
                torch.testing.assert_close(
                    aux_implicit["logits"], aux_explicit["logits"], rtol=0.0, atol=0.0
                )

    def test_regularizer_excludes_padding_positions(self):
        valid, x_a, _, key_padding_mask = self._inputs()
        for ste_cls in (HardSTE, SoftSTE):
            with self.subTest(ste=ste_cls.__module__):
                model = self._make_ste(ste_cls, regularized=True).train()
                _, short_aux = model(valid, layer_idx=0)
                _, padded_aux = model(
                    x_a, layer_idx=0, src_key_padding_mask=key_padding_mask
                )
                torch.testing.assert_close(
                    short_aux["reg_loss"], padded_aux["reg_loss"], rtol=0.0, atol=1e-7
                )

    def test_invalid_mask_contract_fails_fast(self):
        _, x_a, _, key_padding_mask = self._inputs()
        for ste_cls in (HardSTE, SoftSTE):
            with self.subTest(ste=ste_cls.__module__):
                model = self._make_ste(ste_cls).eval()
                with self.assertRaises(TypeError):
                    model(
                        x_a,
                        layer_idx=0,
                        src_key_padding_mask=key_padding_mask.float(),
                    )
                with self.assertRaises(ValueError):
                    model(
                        x_a,
                        layer_idx=0,
                        src_key_padding_mask=key_padding_mask[:, :-1],
                    )


if __name__ == "__main__":
    unittest.main()
