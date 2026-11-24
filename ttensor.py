"""Typed tensor demo. Check with `pyright ttensor.py`; see test_ttensor.py for usage."""
# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false
# pyright: reportInvalidTypeVarUse=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self, get_args, get_origin, overload
from typing import Literal as L

import torch

Device = L["cpu", "cuda"]
DType = float | int | bool | complex


class Dim[*Ts]:
    """Shape marker, e.g. Dim[L[2], L[3]] or Dim[L["batch"], L[3]]. Any types allowed."""


UnknownShape = Dim[*tuple[Any, ...]]

type Unary[S: Dim, D: Device, T: DType] = Callable[[], TTensor[S, D, T]]


class BinaryOp[S: Dim, D: Device, T: DType, R: DType = T](Protocol):
    """Same shape or a scalar keeps the shape; any other TTensor gives UnknownShape."""

    @overload
    def __call__(self, other: TTensor[S, D, T], /) -> TTensor[S, D, R]: ...
    @overload
    def __call__(self, other: DType, /) -> TTensor[S, D, R]: ...
    @overload
    def __call__(self, other: TTensor[UnknownShape, D, T], /) -> TTensor[UnknownShape, D, R]: ...


class Reduction[D: Device, R: DType](Protocol):
    @overload
    def __call__(self) -> TTensor[Dim[()], D, R]: ...
    @overload
    def __call__(self, dim: int | Sequence[int], keepdim: bool = False) -> TTensor[UnknownShape, D, R]: ...


class Extremum[D: Device, T: DType](Protocol):
    @overload
    def __call__(self) -> TTensor[Dim[()], D, T]: ...
    @overload
    def __call__(
        self, dim: int, keepdim: bool = False
    ) -> tuple[TTensor[UnknownShape, D, T], TTensor[UnknownShape, D, int]]: ...


class TTensor[TShape: Dim = UnknownShape, TDevice: Device = Any, TDType: DType = Any](torch.Tensor):
    @classmethod
    def wrap(cls, t: torch.Tensor) -> TTensor[TShape, TDevice, TDType]:
        return t.as_subclass(cls)

    def as_shape[*Ns](self, shape: type[Dim[*Ns]]) -> TTensor[Dim[*Ns], TDevice, TDType]:
        """Unsafe, trusted cast: assigns `shape` without checking it."""
        return self  # type: ignore

    # device and dtype
    def cpu(self) -> TTensor[TShape, L["cpu"], TDType]:
        return super().cpu()  # type: ignore

    def cuda(self) -> TTensor[TShape, L["cuda"], TDType]:
        return super().cuda()  # type: ignore

    @overload
    def to[D: DType](self, dtype: type[D]) -> TTensor[TShape, TDevice, D]: ...
    @overload
    def to[Dv: Device](self, device: Dv) -> TTensor[TShape, Dv, TDType]: ...
    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def item(self) -> TDType:
        return super().item()  # type: ignore

    # elementwise, shape preserving
    abs: Unary[TShape, TDevice, TDType]
    neg: Unary[TShape, TDevice, TDType]
    exp: Unary[TShape, TDevice, TDType]
    log: Unary[TShape, TDevice, TDType]
    sqrt: Unary[TShape, TDevice, TDType]
    rsqrt: Unary[TShape, TDevice, TDType]
    sin: Unary[TShape, TDevice, TDType]
    cos: Unary[TShape, TDevice, TDType]
    tanh: Unary[TShape, TDevice, TDType]
    sigmoid: Unary[TShape, TDevice, TDType]
    relu: Unary[TShape, TDevice, TDType]
    floor: Unary[TShape, TDevice, TDType]
    ceil: Unary[TShape, TDevice, TDType]
    square: Unary[TShape, TDevice, TDType]
    sign: Unary[TShape, TDevice, TDType]
    clone: Unary[TShape, TDevice, TDType]
    detach: Unary[TShape, TDevice, TDType]
    contiguous: Unary[TShape, TDevice, TDType]
    __neg__: Unary[TShape, TDevice, TDType]

    def __abs__(self) -> Self:  # builtin abs() needs a real method, not a declared callable
        return super().__abs__()  # type: ignore

    def clamp(self, min: float | None = None, max: float | None = None) -> Self:
        return super().clamp(min, max)  # type: ignore

    def softmax(self, dim: int) -> Self:
        return super().softmax(dim)  # type: ignore

    def log_softmax(self, dim: int) -> Self:
        return super().log_softmax(dim)  # type: ignore

    def cumsum(self, dim: int) -> Self:
        return super().cumsum(dim)  # type: ignore

    def flip(self, *dims: int) -> Self:
        return super().flip(*dims)  # type: ignore

    def masked_fill(self, mask: TTensor[Any, TDevice, bool], value: DType) -> Self:
        return super().masked_fill(mask, value)  # type: ignore

    # elementwise, binary
    __add__: BinaryOp[TShape, TDevice, TDType]
    __radd__: BinaryOp[TShape, TDevice, TDType]
    __sub__: BinaryOp[TShape, TDevice, TDType]
    __rsub__: BinaryOp[TShape, TDevice, TDType]
    __mul__: BinaryOp[TShape, TDevice, TDType]
    __rmul__: BinaryOp[TShape, TDevice, TDType]
    __truediv__: BinaryOp[TShape, TDevice, TDType]
    __rtruediv__: BinaryOp[TShape, TDevice, TDType]
    __pow__: BinaryOp[TShape, TDevice, TDType]
    __rpow__: BinaryOp[TShape, TDevice, TDType]
    add: BinaryOp[TShape, TDevice, TDType]
    sub: BinaryOp[TShape, TDevice, TDType]
    mul: BinaryOp[TShape, TDevice, TDType]
    div: BinaryOp[TShape, TDevice, TDType]
    pow: BinaryOp[TShape, TDevice, TDType]
    maximum: BinaryOp[TShape, TDevice, TDType]
    minimum: BinaryOp[TShape, TDevice, TDType]
    __lt__: BinaryOp[TShape, TDevice, TDType, bool]
    __le__: BinaryOp[TShape, TDevice, TDType, bool]
    __gt__: BinaryOp[TShape, TDevice, TDType, bool]
    __ge__: BinaryOp[TShape, TDevice, TDType, bool]
    eq: BinaryOp[TShape, TDevice, TDType, bool]
    ne: BinaryOp[TShape, TDevice, TDType, bool]

    # reductions
    sum: Reduction[TDevice, TDType]
    mean: Reduction[TDevice, TDType]
    prod: Reduction[TDevice, TDType]
    std: Reduction[TDevice, TDType]
    var: Reduction[TDevice, TDType]
    amax: Reduction[TDevice, TDType]
    amin: Reduction[TDevice, TDType]
    argmax: Reduction[TDevice, int]
    argmin: Reduction[TDevice, int]
    all: Reduction[TDevice, bool]
    any: Reduction[TDevice, bool]
    max: Extremum[TDevice, TDType]
    min: Extremum[TDevice, TDType]

    # linear algebra
    def matmul[*Ts, M, K, N](
        self: TTensor[Dim[*Ts, M, K], TDevice, TDType],
        other: TTensor[Dim[*Ts, K, N], TDevice, TDType],
    ) -> TTensor[Dim[*Ts, M, N], TDevice, TDType]:
        return super().matmul(other)  # type: ignore

    def __matmul__[*Ts, M, K, N](
        self: TTensor[Dim[*Ts, M, K], TDevice, TDType],
        other: TTensor[Dim[*Ts, K, N], TDevice, TDType],
    ) -> TTensor[Dim[*Ts, M, N], TDevice, TDType]:
        return self.matmul(other)

    def __rmatmul__[*Ts, M, K, N](
        self: TTensor[Dim[*Ts, K, N], TDevice, TDType],
        other: TTensor[Dim[*Ts, M, K], TDevice, TDType],
    ) -> TTensor[Dim[*Ts, M, N], TDevice, TDType]:
        return other.matmul(self)

    def mm[M, K, N](
        self: TTensor[Dim[M, K], TDevice, TDType], mat2: TTensor[Dim[K, N], TDevice, TDType]
    ) -> TTensor[Dim[M, N], TDevice, TDType]:
        return super().mm(mat2)  # type: ignore

    def bmm[B, M, K, N](
        self: TTensor[Dim[B, M, K], TDevice, TDType], mat2: TTensor[Dim[B, K, N], TDevice, TDType]
    ) -> TTensor[Dim[B, M, N], TDevice, TDType]:
        return super().bmm(mat2)  # type: ignore

    def mv[M, K](
        self: TTensor[Dim[M, K], TDevice, TDType], vec: TTensor[Dim[K], TDevice, TDType]
    ) -> TTensor[Dim[M], TDevice, TDType]:
        return super().mv(vec)  # type: ignore

    def dot[K](
        self: TTensor[Dim[K], TDevice, TDType], other: TTensor[Dim[K], TDevice, TDType]
    ) -> TTensor[Dim[()], TDevice, TDType]:
        return super().dot(other)  # type: ignore

    # transposition
    def t[M, N](
        self: TTensor[Dim[M, N], TDevice, TDType],
    ) -> TTensor[Dim[N, M], TDevice, TDType]:
        return super().t()  # type: ignore

    @property
    def mT[*Ts, M, N](
        self: TTensor[Dim[*Ts, M, N], TDevice, TDType],
    ) -> TTensor[Dim[*Ts, N, M], TDevice, TDType]:
        return super().mT  # type: ignore

    @overload
    def transpose[*Ts, M, N](
        self: TTensor[Dim[*Ts, M, N], TDevice, TDType], dim0: L[-2], dim1: L[-1]
    ) -> TTensor[Dim[*Ts, N, M], TDevice, TDType]: ...
    @overload
    def transpose[*Ts, M, N](
        self: TTensor[Dim[*Ts, M, N], TDevice, TDType], dim0: L[-1], dim1: L[-2]
    ) -> TTensor[Dim[*Ts, N, M], TDevice, TDType]: ...
    @overload
    def transpose(self, dim0: int, dim1: int) -> TTensor[UnknownShape, TDevice, TDType]: ...
    def transpose(self, dim0, dim1):
        return super().transpose(dim0, dim1)

    def permute(self, *dims: int) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().permute(*dims)  # type: ignore

    # reshaping: shapes are unknown unless they are trivially derivable
    def view(self, *shape: int | Sequence[int]) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().view(*shape)  # type: ignore

    def reshape(self, *shape: int | Sequence[int]) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().reshape(*shape)  # type: ignore

    def view_as[S: Dim](self, other: TTensor[S, TDevice, TDType]) -> TTensor[S, TDevice, TDType]:
        return super().view_as(other)  # type: ignore

    def reshape_as[S: Dim](self, other: TTensor[S, TDevice, TDType]) -> TTensor[S, TDevice, TDType]:
        return super().reshape_as(other)  # type: ignore

    def expand_as[S: Dim](self, other: TTensor[S, TDevice, TDType]) -> TTensor[S, TDevice, TDType]:
        return super().expand_as(other)  # type: ignore

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().flatten(start_dim, end_dim)  # type: ignore

    def unflatten(self, dim: int, sizes: Sequence[int]) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().unflatten(dim, sizes)  # type: ignore

    def expand(self, *sizes: int | Sequence[int]) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().expand(*sizes)  # type: ignore

    def repeat(self, *sizes: int | Sequence[int]) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().repeat(*sizes)  # type: ignore

    @overload
    def squeeze[*Ts](
        self: TTensor[Dim[L[1], *Ts], TDevice, TDType], dim: L[0]
    ) -> TTensor[Dim[*Ts], TDevice, TDType]: ...
    @overload
    def squeeze[*Ts](
        self: TTensor[Dim[*Ts, L[1]], TDevice, TDType], dim: L[-1]
    ) -> TTensor[Dim[*Ts], TDevice, TDType]: ...
    @overload
    def squeeze(self, dim: int | None = None) -> TTensor[UnknownShape, TDevice, TDType]: ...
    def squeeze(self, dim=None):
        return super().squeeze() if dim is None else super().squeeze(dim)

    @overload
    def unsqueeze[*Ts](
        self: TTensor[Dim[*Ts], TDevice, TDType], dim: L[0]
    ) -> TTensor[Dim[L[1], *Ts], TDevice, TDType]: ...
    @overload
    def unsqueeze[*Ts](
        self: TTensor[Dim[*Ts], TDevice, TDType], dim: L[-1]
    ) -> TTensor[Dim[*Ts, L[1]], TDevice, TDType]: ...
    @overload
    def unsqueeze(self, dim: int) -> TTensor[UnknownShape, TDevice, TDType]: ...
    def unsqueeze(self, dim):
        return super().unsqueeze(dim)

    # indexing and splitting
    def __getitem__(self, index: Any) -> TTensor[UnknownShape, TDevice, TDType]:
        return super().__getitem__(index)  # type: ignore

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TTensor[UnknownShape, TDevice, TDType], ...]:
        return super().chunk(chunks, dim)  # type: ignore

    def split(
        self, split_size: int | Sequence[int], dim: int = 0
    ) -> tuple[TTensor[UnknownShape, TDevice, TDType], ...]:
        return super().split(split_size, dim)  # type: ignore

    def unbind(self, dim: int = 0) -> tuple[TTensor[UnknownShape, TDevice, TDType], ...]:
        return super().unbind(dim)  # type: ignore


def cat[D: Device, T: DType](
    tensors: Sequence[TTensor[Any, D, T]], dim: int = 0
) -> TTensor[UnknownShape, D, T]:
    return TTensor.wrap(torch.cat(tuple(tensors), dim))


def stack[D: Device, T: DType](
    tensors: Sequence[TTensor[Any, D, T]], dim: int = 0
) -> TTensor[UnknownShape, D, T]:
    return TTensor.wrap(torch.stack(tuple(tensors), dim))


# construction: pass a shape type, `zeros(Dim[L[2], L[3]])`, or sizes, `zeros(2, 3)`
class Factory(Protocol):
    @overload
    def __call__[*Ns, Dv: Device = L["cpu"], T: DType = float](
        self, shape: type[Dim[*Ns]], /, *, device: Dv = ..., dtype: type[T] = ...
    ) -> TTensor[Dim[*Ns], Dv, T]: ...
    @overload
    def __call__[Dv: Device = L["cpu"], T: DType = float](
        self, *size: int, device: Dv = ..., dtype: type[T] = ...
    ) -> TTensor[UnknownShape, Dv, T]: ...


class Like(Protocol):
    def __call__[S: Dim, D: Device, T: DType](self, tensor: TTensor[S, D, T], /) -> TTensor[S, D, T]: ...


def _factory(fn: Callable[..., torch.Tensor]) -> Factory:
    def make(*size: Any, device: str = "cpu", dtype: Any = None) -> Any:
        if size and get_origin(size[0]) is Dim:
            size = tuple(get_args(d)[0] for d in get_args(size[0]))
        return TTensor.wrap(fn(*size, device=device, dtype=dtype))

    return make


def _like(fn: Callable[[torch.Tensor], torch.Tensor]) -> Like:
    def make(tensor: Any) -> Any:
        return TTensor.wrap(fn(tensor))

    return make


zeros = _factory(torch.zeros)
ones = _factory(torch.ones)
empty = _factory(torch.empty)
rand = _factory(torch.rand)
randn = _factory(torch.randn)

zeros_like = _like(torch.zeros_like)
ones_like = _like(torch.ones_like)
rand_like = _like(torch.rand_like)
randn_like = _like(torch.randn_like)