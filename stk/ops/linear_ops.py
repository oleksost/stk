import torch

from stk.backend import sputnik
from stk.matrix import Matrix


def dsd(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    return sputnik.dsd(
        a.size(),
        a.data,
        a.offsets,
        a.row_indices,
        a.column_indices,
        a.offsets_t,
        a.column_indices_t,
        a.block_offsets_t,
        not a.is_contiguous(),
        b,
    )


def dds(a, b):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, Matrix)
    return sputnik.dds(
        a,
        b.size(),
        b.data,
        b.offsets,
        b.row_indices,
        b.column_indices,
        b.offsets_t,
        b.column_indices_t,
        b.block_offsets_t,
        not b.is_contiguous(),
    )


def sdd(a, b, topo):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous()
    out = sputnik.sdd(
        a,
        b,
        topo.size(),
        topo.data,
        topo.offsets,
        topo.row_indices,
        topo.column_indices,
        topo.offsets_t,
        topo.column_indices_t,
        topo.block_offsets_t,
    )
    return Matrix(
        topo.size(),
        out,
        topo.row_indices,
        topo.column_indices,
        topo.offsets,
        topo.column_indices_t,
        topo.offsets_t,
        topo.block_offsets_t,
    )


def sdd_adamerge(a, b, out_topo: Matrix, out_adaps: Matrix, layout):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(out_topo, Matrix)
    assert out_topo.is_contiguous()
    assert isinstance(out_adaps, Matrix)
    assert out_adaps.data.is_contiguous()
    assert isinstance(layout, torch.Tensor)
    assert layout.is_contiguous()
    # essentially merged the adapters into a single Matrix()

    out = sputnik.sdd_spsmerge(
        a,
        b,
        out_topo.size(),
        out_topo.data,
        out_topo.row_indices,
        out_topo.column_indices,
        out_topo.column_indices_t,
        out_topo.block_offsets_t,
        out_adaps.data,
        layout,
    )
    return Matrix(
        out_topo.size(),
        out,
        out_topo.row_indices,
        out_topo.column_indices,
        out_topo.offsets,
        out_topo.column_indices_t,
        out_topo.offsets_t,
        out_topo.block_offsets_t,
    )
