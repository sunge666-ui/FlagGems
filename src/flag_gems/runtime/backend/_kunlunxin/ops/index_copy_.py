# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


# Kunlunxin override for index_copy / index_copy_.
#
# The previous rank1/2/3 kernels loaded the dim index as a vector
# (`tl.load(index + index_coord)`), where index_coord for dim=0 changes only
# every `inner`-th lane. Triton XPU mis-compiles that "mostly-constant vector
# gather" and returns stale values at the block transition, producing wrong
# results at the first few columns of every row (see the accuracy bug record).
# The fix moves the dim index into grid axis 1 so `tl.load(index + i)` is a
# scalar load, which sidesteps the buggy vectorized gather entirely.


@libentry()
@triton.jit
def _index_copy_scalar_kernel(
    index,
    src,
    out,
    outer,
    src_dim,
    out_dim,
    inner,
    BLOCK: tl.constexpr,
):
    # grid = (cdiv(outer * inner, BLOCK), src_dim)
    pid_w = tl.program_id(0)  # chunk of (outer * inner)
    i = tl.program_id(1)  # scalar dim index
    w = pid_w * BLOCK + tl.arange(0, BLOCK)
    mask = w < outer * inner
    o = w // inner
    inner_off = w % inner
    src_flat = (o * src_dim + i) * inner + inner_off
    val = tl.load(src + src_flat, mask=mask, other=0.0)
    dst_k = tl.load(index + i)  # scalar load, avoids the vectorized gather bug
    dst_flat = (o * out_dim + dst_k) * inner + inner_off
    tl.store(out + dst_flat, val, mask=mask)


def _validate(inp, dim, index, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    assert bool(
        ((0 <= index) & (index < inp.size(dim))).all()
    ), "0 <= index < self.size(dim)"


def _launch(inp, dim, index, src):
    outer = math.prod(inp.shape[:dim])
    inner = math.prod(inp.shape[dim + 1 :])
    src_dim = src.size(dim)
    out_dim = inp.size(dim)
    block = 4096
    grid = (triton.cdiv(outer * inner, block), src_dim)
    _index_copy_scalar_kernel[grid](
        index,
        src,
        inp,
        outer,
        src_dim,
        out_dim,
        inner,
        BLOCK=block,
        num_warps=8,
    )


def index_copy_(inp, dim, index, src):
    dim %= inp.ndim
    _validate(inp, dim, index, src)
    src = src.contiguous()
    index = index.contiguous()
    if not inp.is_contiguous():
        # in-place on a non-contiguous input: run on a contiguous copy and write
        # the result back (the scalar kernel assumes contiguous flat layout).
        inp_c = inp.contiguous()
        _launch(inp_c, dim, index, src)
        inp.copy_(inp_c)
        return inp
    _launch(inp, dim, index, src)
    return inp


def index_copy(inp, dim, index, src):
    dim %= inp.ndim
    _validate(inp, dim, index, src)
    out = inp.clone()
    return index_copy_(out, dim, index, src)
