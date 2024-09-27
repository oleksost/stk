import itertools
import os
import unittest

import numpy as np
import torch
from absl.testing import parameterized

# os.environ["TRITON_INTERPRET"] = "1"
import stk
from stk.matrix import Matrix

blocksize = 16


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


# An assortment of problems designed to make sure
# the bindings are operating correctly.
_MATRIX_SIZES = (
    (128, 128, 128, 0.95),
    # (128, 128, 128, 0.0),
    # (256, 256, 256, 0.5),
    # (2048, 1024, 512, 0.8),
    # (512, 128, 128, 0.0),
    # (128, 128, 512, 0.0),
    # (1024, 512, 512, 0.0),
    # (1024, 512, 512, 0.5),
    # (1024, 512, 512, 0.75),
    # (512, 512, 1024, 0.0),
    # (512, 512, 1024, 0.5),
    # (512, 512, 1024, 0.75),
    # (1024, 1024, 1024, 0.0),
    # (1024, 1024, 1024, 0.5),
    # (1024, 1024, 1024, 0.75),
)

_TRANSPOSE = (
    (False, False),
    # (False, True),
    # (True, False),
    # (True, True),
)

_DTYPE = (torch.float16, torch.bfloat16)


def _generate_testcases():
    testcases = itertools.product(_MATRIX_SIZES, _TRANSPOSE, _DTYPE)
    testcases = [
        (*size, *trans, blocksize, dtype) for (size, trans, dtype) in testcases
    ]
    return testcases


_LINEAR_OP_TESTS = _generate_testcases()


def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (
        dense.to(cuda_device).requires_grad_(True),
        sparse.to(cuda_device).requires_grad_(True),
    )


def _dense(rows, cols, dtype, std=0.1):
    cuda_device = torch.device("cuda")
    out = (torch.randn(rows, cols) * std).type(dtype)
    return out.to(cuda_device).requires_grad_(True)


def _dense_2x(rows, cols, dtype):
    a = _dense(rows, cols, dtype)
    return a, a.detach().requires_grad_(True)


def _with_transpose(op, a, b, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b)


def _mmm(a, b, topo):
    mask = stk.ops.to_dense(stk.ops.ones_like(topo))
    return torch.mm(a, b) * mask


def _mmm_with_adapters(a, b, topo, adapters):
    adaps_as_dense = [stk.ops.to_dense(adap) for adap in adapters]
    b = b + torch.cat(adaps_as_dense, dim=1)
    mask = stk.ops.to_dense(stk.ops.ones_like(topo))
    return torch.mm(a, b) * mask


def _sparse_out_with_transpose(op, a, b, topo, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b, topo)


def _mask(x, mask):
    mask = stk.ops.to_dense(stk.ops.ones_like(mask))
    return x * mask


@parameterized.parameters(*_LINEAR_OP_TESTS)
class LinearOpsTest(parameterized.TestCase):

    def testLinearOps_Dsd(self, m, k, n, sparsity, trans_a, trans_b, blocking, dtype):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking, dtype)
        b_shape = (n, k) if trans_b else (k, n)
        b, bcp = _dense_2x(*b_shape, dtype)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dsd, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, a_dense, bcp, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        out.sum().backward()

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = stk.ops.to_dense(a.grad)
        expected_grad = _mask(a_dense.grad, a.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = b.grad
        expected_grad = bcp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Dds(self, m, k, n, sparsity, trans_a, trans_b, blocking, dtype):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape, dtype)
        b_shape = (n, k) if trans_b else (k, n)
        b_dense, b = _dense_and_sparse(*b_shape, sparsity, blocking, dtype)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dds, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, acp, b_dense, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        out.sum().backward()

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = a.grad
        expected_grad = acp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = stk.ops.to_dense(b.grad)
        expected_grad = _mask(b_dense.grad, b.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Sdd_wAdapters(
        self, m, k, n, sparsity, trans_a, trans_b, blocking, dtype
    ):
        if trans_a or trans_b:
            return
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape, dtype)

        n_adaps = 10
        adapters = [
            _dense_and_sparse(*(k, n), sparsity, blocking, dtype)[1]
            for _ in range(n_adaps)
        ]
        # merge all adapters into a single sparse Matrix()
        adaps: Matrix = stk.ops.merge_adapters(adapters)

        out_shape = (m, n * n_adaps)
        _, out_topo = _dense_and_sparse(*out_shape, sparsity, blocking, dtype)
        # create a mapping from out_topo to adaps, indicating whether each out_topo bvlock needs to be merged with an adapter block, and if so which one
        layout = stk.ops.create_ada_layout(adaps)

        b_shape = (n * n_adaps, k) if trans_b else (k, n * n_adaps)
        b, bcp = _dense_2x(*b_shape, dtype)
        # Execute the matmul.
        out = stk.ops.sdd_adamerge(a, b, out_topo, adaps, layout)
        expected_out = _mmm_with_adapters(acp, bcp, out_topo, adapters)
        
        
        adapters_as_dense = torch.cat([stk.ops.to_dense(adap) for adap in adapters], dim=1)
        adaps_as_dense = stk.ops.to_dense(adaps)
        assert torch.sum(adapters_as_dense!=adaps_as_dense)==0, "adapters and adaps should be the same"
        
        # Validate the results.
        out = stk.ops.to_dense(out)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

    def testLinearOps_Sdd(self, m, k, n, sparsity, trans_a, trans_b, blocking, dtype):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape, dtype)
        b_shape = (n, k) if trans_b else (k, n)
        b, bcp = _dense_2x(*b_shape, dtype)
        _, topo = _dense_and_sparse(m, n, sparsity, blocking, dtype)

        # Execute the matmul.
        out = _sparse_out_with_transpose(stk.ops.sdd, a, b, topo, trans_a, trans_b)
        expected_out = _sparse_out_with_transpose(_mmm, acp, bcp, topo, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        stk.ops.sum(out).backward()

        # Validate the results.
        out = stk.ops.to_dense(out)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = a.grad
        expected_grad = acp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = b.grad
        expected_grad = bcp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))


if __name__ == "__main__":
    unittest.main()
