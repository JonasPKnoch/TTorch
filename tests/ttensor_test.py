"""Compare TTensor with native torch, at runtime (pytest) and statically (pyright).

Run: `pytest test_ttensor.py && pyright test_ttensor.py`

Expected type errors are marked `# pyright: ignore[rule]`. With
`reportUnnecessaryTypeIgnoreComment` enabled, pyright fails if a marked line is
*not* an error, so the file checks cleanly only when the errors are exactly these.
"""
# pyright: reportUnnecessaryTypeIgnoreComment=true

import operator
from typing import Any
from typing import Literal as L
from typing import assert_type

import pytest
import torch

from ttensor import (
    Concatenated,
    Dim,
    Flattened,
    Indexed,
    Permuted,
    Reshaped,
    TTensor,
    UnknownShape,
    cat,
    empty,
    ones,
    ones_like,
    rand,
    rand_like,
    randn,
    randn_like,
    stack,
    zeros,
    zeros_like,
)

Cpu = L["cpu"]
Unk = TTensor[UnknownShape, Cpu, float]
F0 = TTensor[Dim[()], Cpu, float]
F13 = TTensor[Dim[L[1], L[3]], Cpu, float]
F23 = TTensor[Dim[L[2], L[3]], Cpu, float]
F24 = TTensor[Dim[L[2], L[4]], Cpu, float]
F32 = TTensor[Dim[L[3], L[2]], Cpu, float]
F34 = TTensor[Dim[L[3], L[4]], Cpu, float]
F45 = TTensor[Dim[L[4], L[5]], Cpu, float]
F64 = TTensor[Dim[L[6], L[4]], Cpu, float]
F234 = TTensor[Dim[L[2], L[3], L[4]], Cpu, float]
F231 = TTensor[Dim[L[2], L[3], L[1]], Cpu, float]
B23 = TTensor[Dim[L[2], L[3]], Cpu, bool]
I23 = TTensor[Dim[L[2], L[3]], Cpu, int]
I34 = TTensor[Dim[L[3], L[4]], Cpu, int]
FLAT234 = TTensor[Dim[Flattened[L[2], L[3], L[4]]], Cpu, float]
R23 = TTensor[Reshaped[Dim[L[2], L[3]]], Cpu, float]
R234 = TTensor[Reshaped[Dim[L[2], L[3], L[4]]], Cpu, float]


def pos(*shape: int) -> torch.Tensor:
    return torch.rand(*shape) + 0.5  # positive, so log/sqrt/pow stay finite


def same(result: Any, expected: torch.Tensor) -> None:
    assert isinstance(result, TTensor)
    assert not isinstance(expected, TTensor)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert torch.equal(result, expected)


def same_all(results: Any, expected: Any) -> None:
    assert len(results) == len(expected)
    for r, e in zip(results, expected):
        same(r, e)


# --- matmul family ---------------------------------------------------------


def test_matmul() -> None:
    x, y = torch.randn(2, 3), torch.randn(3, 4)
    a, b = F23.wrap(x), F34.wrap(y)

    same(a.matmul(b), x.matmul(y))
    same(a @ b, x @ y)
    assert_type(a @ b, F24)


def test_batched_matmul() -> None:
    x, y = torch.randn(8, 2, 3), torch.randn(8, 3, 4)
    a = TTensor[Dim[L["batch"], L[2], L[3]], Cpu, float].wrap(x)
    b = TTensor[Dim[L["batch"], L[3], L[4]], Cpu, float].wrap(y)

    same(a @ b, x @ y)
    assert_type(a @ b, TTensor[Dim[L["batch"], L[2], L[4]], Cpu, float])


def test_mm_bmm_mv_dot() -> None:
    x, y, v = torch.randn(2, 3), torch.randn(3, 4), torch.randn(3)
    a, b, va = F23.wrap(x), F34.wrap(y), TTensor[Dim[L[3]], Cpu, float].wrap(v)
    bx, by = torch.randn(8, 2, 3), torch.randn(8, 3, 4)
    ba = TTensor[Dim[L[8], L[2], L[3]], Cpu, float].wrap(bx)
    bb = TTensor[Dim[L[8], L[3], L[4]], Cpu, float].wrap(by)

    same(a.mm(b), x.mm(y))
    assert_type(a.mm(b), F24)
    same(ba.bmm(bb), bx.bmm(by))
    assert_type(ba.bmm(bb), TTensor[Dim[L[8], L[2], L[4]], Cpu, float])
    same(a.mv(va), x.mv(v))
    assert_type(a.mv(va), TTensor[Dim[L[2]], Cpu, float])
    same(va.dot(va), v.dot(v))
    assert_type(va.dot(va), F0)


def test_transpose() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    same(a.t(), x.t())
    assert_type(a.t(), F32)
    same(a.mT, x.mT)
    assert_type(a.mT, F32)
    same(a.transpose(-2, -1), x.transpose(-2, -1))
    assert_type(a.transpose(-2, -1), F32)
    same(a.transpose(-1, -2), x.transpose(-1, -2))
    assert_type(a.transpose(-1, -2), F32)
    same(a.transpose(0, 1), x.transpose(0, 1))
    assert_type(a.transpose(0, 1), Unk)  # not special-cased


def test_unspecified_tensor_accepts_anything() -> None:
    x, y = torch.randn(2, 3), torch.randn(3, 4)
    u = TTensor.wrap(x)  # all type parameters left at their defaults
    b = F34.wrap(y)

    same(u @ b, x @ y)
    same(b.t() @ u.t(), y.t() @ x.t())
    same(u + u, x + x)


# --- elementwise -----------------------------------------------------------

UNARY = "abs neg exp log sqrt rsqrt sin cos tanh sigmoid relu floor ceil square sign clone detach contiguous"
BINARY = "add sub mul div pow maximum minimum eq ne"


@pytest.mark.parametrize("name", UNARY.split())
def test_unary_ops(name: str) -> None:
    x = pos(2, 3)
    same(getattr(F23.wrap(x), name)(), getattr(x, name)())


@pytest.mark.parametrize("name", BINARY.split())
def test_binary_methods(name: str) -> None:
    x, y = pos(2, 3), pos(2, 3)
    a, b = F23.wrap(x), F23.wrap(y)
    same(getattr(a, name)(b), getattr(x, name)(y))


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
    + [operator.lt, operator.le, operator.gt, operator.ge],
)
def test_operators(op: Any) -> None:
    x, y = pos(2, 3), pos(2, 3)
    a, b = F23.wrap(x), F23.wrap(y)

    same(op(a, b), op(x, y))
    same(op(a, 1.5), op(x, 1.5))
    same(op(1.5, a), op(1.5, x))


def test_elementwise_types() -> None:
    a = F23.wrap(pos(2, 3))
    b = F23.wrap(pos(2, 3))

    assert_type(a.exp(), F23)
    assert_type(-a, F23)
    assert_type(abs(a), F23)
    assert_type(a + b, F23)
    assert_type(a - 1, F23)
    assert_type(2 * a, F23)
    assert_type(a / b, F23)
    assert_type(a**2, F23)
    assert_type(a.maximum(b), F23)
    assert_type(a < b, B23)
    assert_type(1.0 >= a, B23)
    assert_type(a.eq(b), B23)


def test_shape_preserving_methods() -> None:
    x = pos(2, 3)
    a = F23.wrap(x)
    m = B23.wrap(x > 1.0)

    same(a.clamp(0.6, 1.0), x.clamp(0.6, 1.0))
    same(a.softmax(-1), x.softmax(-1))
    same(a.log_softmax(0), x.log_softmax(0))
    same(a.cumsum(1), x.cumsum(1))
    same(a.flip(0, 1), x.flip(0, 1))
    same(a.masked_fill(m, 0.0), x.masked_fill(x > 1.0, 0.0))
    assert_type(a.clamp(0.6, 1.0), F23)
    assert_type(a.softmax(-1), F23)
    assert_type(a.masked_fill(m, 0.0), F23)


def test_broadcast_falls_back_to_unknown_shape() -> None:
    x, y = torch.randn(2, 3), torch.randn(1, 3)
    a, b = F23.wrap(x), F13.wrap(y)

    same(a + b, x + y)
    assert_type(a + b, Unk)
    same(a < b, x < y)


# --- reductions ------------------------------------------------------------


@pytest.mark.parametrize("name", "sum mean prod std var amax amin argmax argmin all any".split())
def test_reductions(name: str) -> None:
    x = pos(2, 3)
    a = F23.wrap(x)

    same(getattr(a, name)(), getattr(x, name)())
    same(getattr(a, name)(1), getattr(x, name)(1))
    same(getattr(a, name)(dim=0, keepdim=True), getattr(x, name)(dim=0, keepdim=True))


def test_reduction_types() -> None:
    a = F23.wrap(pos(2, 3))

    assert_type(a.sum(), F0)
    assert_type(a.mean(1), Unk)
    assert_type(a.sum(dim=0, keepdim=True), Unk)
    assert_type(a.argmax(), TTensor[Dim[()], Cpu, int])
    assert_type(a.all(), TTensor[Dim[()], Cpu, bool])
    assert_type(a.sum().item(), float)
    assert a.sum().item() == a.sum().item()


def test_max_min() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    same(a.max(), x.max())
    same(a.min(), x.min())
    assert_type(a.max(), F0)
    values, indices = a.max(1)
    expected = x.max(1)
    same(values, expected.values)
    same(indices, expected.indices)
    assert_type(a.max(1), tuple[Unk, TTensor[UnknownShape, Cpu, int]])


# --- reshaping -------------------------------------------------------------


def test_view_reshape() -> None:
    x, y = torch.randn(2, 3, 4), torch.randn(6, 4)
    a, b = F234.wrap(x), F64.wrap(y)

    same(a.view(6, 4), x.view(6, 4))
    same(a.view((6, -1)), x.view((6, -1)))
    same(a.reshape(6, 4), x.reshape(6, 4))
    assert_type(a.view(6, 4), R234)
    assert_type(a.reshape(6, 4), R234)
    same(a.view_as(b), x.view_as(y))
    same(a.reshape_as(b), x.reshape_as(y))
    assert_type(a.view_as(b), F64)
    assert_type(a.reshape_as(b), F64)


def test_flatten_expand_permute() -> None:
    x, e = torch.randn(2, 3, 4), torch.randn(1, 3)
    a, f = F234.wrap(x), F13.wrap(e)

    same(a.flatten(), x.flatten())
    same(a.flatten(1), x.flatten(1))
    same(a.unflatten(2, (2, 2)), x.unflatten(2, (2, 2)))
    same(a.permute(2, 0, 1), x.permute(2, 0, 1))
    same(f.expand(2, 3), e.expand(2, 3))
    same(f.expand_as(F23.wrap(torch.randn(2, 3))), e.expand_as(torch.randn(2, 3)))
    same(f.repeat(2, 1), e.repeat(2, 1))
    assert_type(a.flatten(), TTensor[Dim[Flattened[L[2], L[3], L[4]]], Cpu, float])
    assert_type(a.permute(2, 0, 1), TTensor[Permuted[Dim[L[2], L[3], L[4]]], Cpu, float])
    assert_type(f.expand(2, 3), Unk)
    assert_type(f.repeat(2, 1), Unk)


def test_squeeze_unsqueeze() -> None:
    x, e, y = torch.randn(2, 3), torch.randn(1, 3), torch.randn(2, 3, 1)
    a, f, g = F23.wrap(x), F13.wrap(e), F231.wrap(y)

    same(f.squeeze(0), e.squeeze(0))
    assert_type(f.squeeze(0), TTensor[Dim[L[3]], Cpu, float])
    same(g.squeeze(-1), y.squeeze(-1))
    assert_type(g.squeeze(-1), F23)
    same(a.squeeze(0), x.squeeze(0))  # not size 1: a no-op in torch, shape unknown for us
    assert_type(a.squeeze(0), Unk)
    same(a.squeeze(), x.squeeze())
    assert_type(a.squeeze(), Unk)

    same(a.unsqueeze(0), x.unsqueeze(0))
    assert_type(a.unsqueeze(0), TTensor[Dim[L[1], L[2], L[3]], Cpu, float])
    same(a.unsqueeze(-1), x.unsqueeze(-1))
    assert_type(a.unsqueeze(-1), F231)
    same(a.unsqueeze(1), x.unsqueeze(1))
    assert_type(a.unsqueeze(1), Unk)


def test_as_shape_is_a_trusted_cast() -> None:
    x, w = torch.randn(2, 3, 4), torch.randn(4, 5)
    a, b = F234.wrap(x), F45.wrap(w)

    r = a.view(6, 4).as_shape(Dim[L[6], L[4]])
    assert_type(r, F64)
    same(r @ b, x.view(6, 4) @ w)
    assert_type(r @ b, TTensor[Dim[L[6], L[5]], Cpu, float])

    wrong = a.as_shape(Dim[L[5], L[5]])  # a false claim is accepted, and never checked
    assert_type(wrong, TTensor[Dim[L[5], L[5]], Cpu, float])
    assert wrong.shape == (2, 3, 4)


# --- indexing, splitting, joining ------------------------------------------


def test_indexing() -> None:
    x = torch.randn(2, 3, 4)
    a = F234.wrap(x)

    same(a[0], x[0])
    same(a[:, 1:], x[:, 1:])
    same(a[a > 0], x[x > 0])
    assert_type(a[0], TTensor[Indexed[Dim[L[2], L[3], L[4]]], Cpu, float])


def test_chunk_split_unbind() -> None:
    x = torch.randn(2, 3, 4)
    a = F234.wrap(x)

    same_all(a.chunk(2, dim=2), x.chunk(2, dim=2))
    same_all(a.split(2, dim=2), x.split(2, dim=2))
    same_all(a.unbind(0), x.unbind(0))
    assert_type(a.chunk(2), tuple[Unk, ...])
    assert_type(a.unbind(0), tuple[Unk, ...])


def test_cat_stack() -> None:
    x, y = torch.randn(2, 3), torch.randn(2, 3)
    a, b = F23.wrap(x), F23.wrap(y)

    same(cat([a, b]), torch.cat([x, y]))
    same(cat([a, b], dim=1), torch.cat([x, y], dim=1))
    same(stack([a, b], dim=1), torch.stack([x, y], dim=1))
    assert_type(cat([a, b]), Unk)
    assert_type(stack([a, b]), Unk)


# --- operation records -----------------------------------------------------


def test_flatten_records_dims() -> None:
    x, y = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    a, b = F234.wrap(x), F234.wrap(y)

    same(a.flatten(), x.flatten())
    assert_type(a.flatten(), FLAT234)
    same(a.flatten(1), x.flatten(1))
    assert_type(a.flatten(1), TTensor[Dim[L[2], Flattened[L[3], L[4]]], Cpu, float])
    same(a.flatten() + b.flatten(), x.flatten() + y.flatten())  # same record, so same shape
    assert_type(a.flatten() + b.flatten(), FLAT234)


def test_flatten_edge_ranks() -> None:
    v, s, m = torch.randn(4), torch.tensor(1.5), torch.randn(2, 3)
    va = TTensor[Dim[L[4]], Cpu, float].wrap(v)
    sa, ma = F0.wrap(s), F23.wrap(m)

    same(va.flatten(), v.flatten())
    assert_type(va.flatten(), TTensor[Dim[L[4]], Cpu, float])  # nothing to record
    same(sa.flatten(), s.flatten())
    assert_type(sa.flatten(), TTensor[Dim[L[1]], Cpu, float])
    same(ma.flatten(1), m.flatten(1))
    assert_type(ma.flatten(1), F23)
    same(ma.flatten(0, 1), m.flatten(0, 1))
    assert_type(ma.flatten(0, 1), Unk)  # other ranges are not tracked


def test_cat_records_dim() -> None:
    x, y = torch.randn(2, 3), torch.randn(2, 3)
    a, b = F23.wrap(x), F23.wrap(y)

    same(cat((a, b)), torch.cat((x, y)))
    assert_type(cat((a, b)), TTensor[Dim[Concatenated[L[2], L[2]], L[3]], Cpu, float])
    same(cat((a, b), dim=-1), torch.cat((x, y), dim=-1))
    assert_type(cat((a, b), dim=-1), TTensor[Dim[L[2], Concatenated[L[3], L[3]]], Cpu, float])
    same(cat([a, b]), torch.cat([x, y]))  # a list has no per-element types
    assert_type(cat([a, b]), Unk)


def test_shape_records_and_depth_cap() -> None:
    x = torch.randn(2, 3, 4)
    a = F234.wrap(x)
    expected = x.view(6, 4).permute(1, 0)[0]

    r = a.view(6, 4)
    assert_type(r, R234)
    p = r.permute(1, 0)
    assert_type(p, TTensor[Permuted[Reshaped[Dim[L[2], L[3], L[4]]]], Cpu, float])
    i = p[0]  # a third record would exceed two levels, so the record restarts
    same(i, expected)
    assert_type(i, TTensor[Indexed[Any], Cpu, float])
    same(i.reshape(2, 3), expected.reshape(2, 3))
    assert_type(i.reshape(2, 3), TTensor[Reshaped[Any], Cpu, float])  # and stays at one level


def test_record_shaped_tensors_compose_with_elementwise_ops() -> None:
    x, z = torch.randn(2, 3, 4), torch.randn(6, 4)
    a, c = F234.wrap(x), F64.wrap(z)
    r, expected = a.view(6, 4), x.view(6, 4)

    same(r + r, expected + expected)
    assert_type(r + r, R234)
    same(r.exp(), expected.exp())
    assert_type(r.exp(), R234)
    same(r.sum(), expected.sum())
    assert_type(r.sum(), F0)
    same(r + c, expected + z)  # a different shape type: falls back to unknown
    assert_type(r + c, Unk)


def test_record_shaped_tensors_do_not_match_shape_patterns() -> None:
    # Known limitation: TShape is invariant, so Reshaped[...] never matches Dim[...].
    x, y = torch.randn(2, 3, 4), torch.randn(4, 5)
    a, w = F234.wrap(x), F45.wrap(y)

    same(a.view(6, 4) @ w, x.view(6, 4) @ y)  # pyright: ignore[reportOperatorIssue]
    same(a.view(6, 4).t(), x.view(6, 4).t())  # pyright: ignore[reportAttributeAccessIssue]

    def takes_unspecified(t: TTensor) -> None: ...

    takes_unspecified(a)
    takes_unspecified(a.flatten())  # dim-level records are fine
    takes_unspecified(a.view(6, 4))  # pyright: ignore[reportArgumentType]


# --- construction ----------------------------------------------------------


def test_zeros_ones() -> None:
    same(zeros(Dim[L[2], L[3]]), torch.zeros(2, 3))
    same(ones(Dim[L[2], L[3]]), torch.ones(2, 3))
    same(zeros(2, 3), torch.zeros(2, 3))
    same(ones(Dim[L[2]], dtype=int), torch.ones(2, dtype=torch.int64))
    assert_type(zeros(Dim[L[2], L[3]]), F23)
    assert_type(zeros(2, 3), Unk)  # sizes are plain ints: shape unknown
    assert_type(ones(Dim[L[2]], dtype=int), TTensor[Dim[L[2]], Cpu, int])
    assert_type(zeros(2, 3, dtype=bool), TTensor[UnknownShape, Cpu, bool])


def test_rand_randn() -> None:
    for make, native in [(rand, torch.rand), (randn, torch.randn)]:
        torch.manual_seed(0)
        expected = native(2, 3)
        torch.manual_seed(0)
        same(make(Dim[L[2], L[3]]), expected)
        torch.manual_seed(0)
        same(make(2, 3), expected)
    assert_type(randn(Dim[L[2], L[3]]), F23)


def test_empty() -> None:
    result, expected = empty(Dim[L[2], L[3]]), torch.empty(2, 3)  # values are arbitrary
    assert isinstance(result, TTensor)
    assert (result.shape, result.dtype) == (expected.shape, expected.dtype)
    assert_type(result, F23)


def test_like() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    same(zeros_like(a), torch.zeros_like(x))
    same(ones_like(a), torch.ones_like(x))
    for make, native in [(rand_like, torch.rand_like), (randn_like, torch.randn_like)]:
        torch.manual_seed(0)
        expected = native(x)
        torch.manual_seed(0)
        same(make(a), expected)
    assert_type(zeros_like(a), F23)
    assert_type(randn_like(a), F23)


# --- device and dtype ------------------------------------------------------


def test_device_dtype_conversion() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    same(a.cpu(), x.cpu())
    assert_type(a.cpu(), F23)
    same(a.to("cpu"), x.to("cpu"))
    assert_type(a.to("cpu"), F23)
    same(a.to(int), x.to(torch.int64))
    assert_type(a.to(int), I23)


# --- errors: torch raises at runtime, pyright rejects statically -----------


def test_matmul_inner_dim_mismatch() -> None:
    x, y = torch.randn(2, 3), torch.randn(4, 5)
    a, b = F23.wrap(x), F45.wrap(y)

    with pytest.raises(RuntimeError):
        _ = x @ y
    with pytest.raises(RuntimeError):
        _ = a @ b  # pyright: ignore[reportOperatorIssue]
    with pytest.raises(RuntimeError):
        _ = a.matmul(b)  # pyright: ignore[reportArgumentType]


def test_matmul_dtype_mismatch() -> None:
    x, y = torch.randn(2, 3), torch.ones(3, 4, dtype=torch.int64)
    a, b = F23.wrap(x), I34.wrap(y)

    with pytest.raises(RuntimeError):
        _ = x @ y
    with pytest.raises(RuntimeError):
        _ = a @ b  # pyright: ignore[reportOperatorIssue]


def test_mm_bmm_mv_dot_mismatch() -> None:
    x, y, v = torch.randn(2, 3), torch.randn(4, 5), torch.randn(4)
    a, b, va = F23.wrap(x), F45.wrap(y), TTensor[Dim[L[4]], Cpu, float].wrap(v)
    bx, by = torch.randn(8, 2, 3), torch.randn(7, 3, 4)
    ba = TTensor[Dim[L[8], L[2], L[3]], Cpu, float].wrap(bx)
    bb = TTensor[Dim[L[7], L[3], L[4]], Cpu, float].wrap(by)
    w = torch.randn(3)
    wa = TTensor[Dim[L[3]], Cpu, float].wrap(w)

    with pytest.raises(RuntimeError):
        _ = x.mm(y)
    with pytest.raises(RuntimeError):
        _ = a.mm(b)  # pyright: ignore[reportArgumentType]
    with pytest.raises(RuntimeError):
        _ = bx.bmm(by)
    with pytest.raises(RuntimeError):
        _ = ba.bmm(bb)  # pyright: ignore[reportArgumentType]
    with pytest.raises(RuntimeError):
        _ = x.mv(v)
    with pytest.raises(RuntimeError):
        _ = a.mv(va)  # pyright: ignore[reportArgumentType]
    with pytest.raises(RuntimeError):
        _ = v.dot(w)
    with pytest.raises(RuntimeError):
        _ = va.dot(wa)  # pyright: ignore[reportArgumentType]


def test_transpose_needs_two_dims() -> None:
    x = torch.randn(2, 3, 4)
    a = F234.wrap(x)

    with pytest.raises(RuntimeError):
        _ = x.t()
    with pytest.raises(RuntimeError):
        _ = a.t()  # pyright: ignore[reportAttributeAccessIssue]


def test_missing_argument() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    with pytest.raises(TypeError):
        _ = x.softmax()  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        _ = a.softmax()  # pyright: ignore[reportCallIssue]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_device_mismatch() -> None:
    x, y = torch.randn(2, 3), torch.randn(3, 4)
    a, b = F23.wrap(x), F34.wrap(y)

    with pytest.raises(RuntimeError):
        _ = x @ y.cuda()
    with pytest.raises(RuntimeError):
        _ = a @ b.cuda()  # pyright: ignore[reportOperatorIssue]


def test_device_mismatch_is_static_error() -> None:
    def check(a: F23, b: F34) -> None:  # never run: torch can't fail here without a GPU
        assert_type(a.cpu() @ b, F24)
        assert_type(a.to("cuda"), TTensor[Dim[L[2], L[3]], L["cuda"], float])
        _ = a @ b.cuda()  # pyright: ignore[reportOperatorIssue]
        _ = a + a.cuda()  # pyright: ignore[reportOperatorIssue]
        _ = cat([a, a.cuda()])  # pyright: ignore[reportCallIssue, reportArgumentType]


# --- known gaps: torch raises, pyright does not (shapes fall back to unknown) ---


def test_add_shape_mismatch_is_not_caught_statically() -> None:
    x, y = torch.randn(2, 3), torch.randn(3, 4)
    a, b = F23.wrap(x), F34.wrap(y)

    with pytest.raises(RuntimeError):
        _ = x + y
    with pytest.raises(RuntimeError):
        assert_type(a + b, Unk)


def test_view_numel_mismatch_is_not_caught_statically() -> None:
    x = torch.randn(2, 3)
    a = F23.wrap(x)

    with pytest.raises(RuntimeError):
        _ = x.view(5, 5)
    with pytest.raises(RuntimeError):
        assert_type(a.view(5, 5), R23)


# --- intentional divergence: TTensor is stricter than torch ----------------


def test_dtype_promotion_is_a_static_error() -> None:
    x, y = torch.randn(2, 3), torch.ones(2, 3, dtype=torch.int64)
    a, b = F23.wrap(x), I23.wrap(y)

    assert (x + y).dtype == torch.float32  # torch promotes int to float
    same(a + b, x + y)  # pyright: ignore[reportOperatorIssue]
    same(cat([a, b]), torch.cat([x, y]))  # pyright: ignore[reportCallIssue, reportArgumentType]