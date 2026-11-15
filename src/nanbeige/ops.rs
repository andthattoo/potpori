//! Typed launches of the kernels in `cu/kernels.cu`. Addresses are raw device pointers so a
//! kernel can write straight into a slice of a larger buffer (a KV cache row, a residual
//! stream); the caller keeps the buffers alive.

use crate::gpu::{gpu, Buf};

use super::config::HEAD_DIM;

/// `out[t, :] = table[ids[t], :]`
pub fn embed(out: u64, table: &Buf<u16>, ids: &Buf<i32>, hidden: usize, s: usize) {
    gpu().launch(
        "embed",
        (hidden.div_ceil(256), s),
        256,
        &[
            out.into(),
            table.ptr().into(),
            ids.ptr().into(),
            hidden.into(),
        ],
    );
}

/// RMSNorm of `rows` rows of `n` values with weight `w`. `out` may equal `x`.
pub fn rmsnorm(out: u64, x: u64, w: &Buf<f32>, eps: f32, n: usize, rows: usize) {
    debug_assert_eq!(w.len(), n);
    gpu().launch(
        "rmsnorm",
        (rows, 1),
        256,
        &[out.into(), x.into(), w.ptr().into(), eps.into(), n.into()],
    );
}

/// Whether a 4-bit product with `n_in` input columns runs on tensor cores, which round each
/// input to bf16 (nearest, ties to even) first: on cards that have them. Products with bf16
/// weights never do: they are bound by reading the weights, and gain nothing.
pub fn rounds_input(n_in: usize) -> bool {
    gpu().tensor_cores() && n_in.is_multiple_of(16)
}

/// `out[t, :n_out] (+)= w x[t, :]` for `s` tokens; `out` rows are `out_stride` long.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    out: u64,
    x: u64,
    w: &Buf<u16>,
    n_out: usize,
    n_in: usize,
    s: usize,
    out_stride: usize,
    add: bool,
) {
    debug_assert_eq!(w.len(), n_out * n_in);
    // The variant that takes the tokens in the fewest groups without padding more than it must.
    let kernel = match s {
        1 => "gemm_t1",
        2 => "gemm_t2",
        3 | 4 => "gemm_t4",
        _ => "gemm_t8",
    };
    gpu().launch(
        kernel,
        (n_out.div_ceil(8), 1),
        128,
        &[
            out.into(),
            x.into(),
            w.ptr().into(),
            n_out.into(),
            n_in.into(),
            s.into(),
            out_stride.into(),
            (add as i32).into(),
        ],
    );
}

/// [`gemm`] with a 4-bit matrix: `w` is `[n_out, n_in / 2]` bytes of two weights each,
/// `scale` is `[n_out, n_in / 16]` 8-bit floats and `global` the matrix's f32 scale.
#[allow(clippy::too_many_arguments)]
pub fn gemm4(
    out: u64,
    x: u64,
    w: &Buf<u8>,
    scale: &Buf<u8>,
    global: f32,
    n_out: usize,
    n_in: usize,
    s: usize,
    out_stride: usize,
    add: bool,
) {
    debug_assert!(n_in.is_multiple_of(16));
    debug_assert_eq!(w.len(), n_out * n_in / 2);
    debug_assert_eq!(scale.len(), n_out * n_in / 16);
    if rounds_input(n_in) {
        return gpu().launch(
            "gemm4_tc",
            (n_out.div_ceil(32), s.div_ceil(8)),
            256,
            &[
                out.into(),
                x.into(),
                w.ptr().into(),
                scale.ptr().into(),
                (global / 2048.0).into(),
                n_out.into(),
                n_in.into(),
                s.into(),
                out_stride.into(),
                (add as i32).into(),
            ],
        );
    }
    let kernel = match s {
        1 => "gemm4_t1",
        2 => "gemm4_t2",
        3 | 4 => "gemm4_t4",
        _ => "gemm4_t8",
    };
    gpu().launch(
        kernel,
        (n_out.div_ceil(8), 1),
        128,
        &[
            out.into(),
            x.into(),
            w.ptr().into(),
            scale.ptr().into(),
            // The kernel decodes weights doubled and scales times 1024.
            (global / 2048.0).into(),
            n_out.into(),
            n_in.into(),
            s.into(),
            out_stride.into(),
            (add as i32).into(),
        ],
    );
}

/// `out[t, i] = silu(gu[t, i]) * gu[t, n + i]`: the nonlinearity of a SwiGLU MLP, after one
/// gemm over the gate and up matrices stacked.
pub fn silu_mul(out: u64, gu: u64, n: usize, s: usize) {
    gpu().launch(
        "silu_mul",
        (n.div_ceil(256), s),
        256,
        &[out.into(), gu.into(), n.into()],
    );
}

/// Rotate `x[s, heads, HEAD_DIM]` in place for positions `pos0..pos0 + s`.
pub fn rope(x: u64, table: &Buf<f32>, heads: usize, pos0: usize, s: usize) {
    debug_assert!((pos0 + s) * HEAD_DIM <= table.len());
    gpu().launch(
        "rope",
        (heads, s),
        HEAD_DIM / 2,
        &[x.into(), table.ptr().into(), heads.into(), pos0.into()],
    );
}

/// Which cached positions each query token sees.
#[derive(Clone, Copy)]
pub enum Span {
    /// Token `t` of a chunk at `pos0` sees `[0, pos0 + t]`.
    Causal { pos0: usize },
    /// Every token sees `[0, len)`.
    All { len: usize },
}

/// Attention of `s` query tokens over a `[position, kv_heads, HEAD_DIM]` cache.
#[allow(clippy::too_many_arguments)]
pub fn attn(
    out: u64,
    q: u64,
    keys: &Buf<f32>,
    values: &Buf<f32>,
    scratch: &Buf<f32>,
    heads: usize,
    kv_heads: usize,
    span: Span,
    cap: usize,
    s: usize,
) {
    let (len0, step) = match span {
        Span::Causal { pos0 } => (pos0 + 1, 1usize),
        Span::All { len } => (len, 0),
    };
    assert!(
        len0 + (s - 1) * step <= cap,
        "attention span exceeds the cache"
    );
    debug_assert!(scratch.len() >= s * heads * cap);
    // Tokens are taken in tiles that share each key and value; see `attn_impl`.
    let (kernel, tile) = match s {
        1 => ("attn_t1", 1),
        2 => ("attn_t2", 2),
        3 | 4 => ("attn_t4", 4),
        _ => ("attn_t8", 8),
    };
    gpu().launch(
        kernel,
        (heads, s.div_ceil(tile)),
        HEAD_DIM,
        &[
            out.into(),
            q.into(),
            keys.ptr().into(),
            values.ptr().into(),
            scratch.ptr().into(),
            heads.into(),
            kv_heads.into(),
            len0.into(),
            step.into(),
            cap.into(),
            (1.0 / (HEAD_DIM as f32).sqrt()).into(),
            s.into(),
        ],
    );
}

/// The drafter's two-tap grouped convolution over `rows` consecutive tokens of one block.
#[allow(clippy::too_many_arguments)]
pub fn conv2(
    out: u64,
    x: u64,
    coef: u64,
    base: &Buf<f32>,
    hidden: usize,
    group: usize,
    side: usize,
    add: bool,
    rows: usize,
) {
    debug_assert_eq!(base.len(), 4 * hidden);
    gpu().launch(
        "conv2",
        (hidden.div_ceil(256), rows),
        256,
        &[
            out.into(),
            x.into(),
            coef.into(),
            base.ptr().into(),
            hidden.into(),
            group.into(),
            side.into(),
            (add as i32).into(),
        ],
    );
}

/// The `k` largest of each of `rows` rows of `vocab` logits, largest first and ties to the
/// lower index: `ids` and `vals` are `[rows, k]`. A row holding a NaN or an infinity gets
/// `ids[row, 0] = -1`.
pub fn topk(ids: &Buf<i32>, vals: &Buf<f32>, logits: u64, vocab: usize, k: usize, rows: usize) {
    debug_assert!(k <= vocab && ids.len() >= rows * k && vals.len() >= rows * k);
    gpu().launch(
        "topk",
        (rows, 1),
        1024,
        &[
            ids.ptr().into(),
            vals.ptr().into(),
            logits.into(),
            vocab.into(),
            k.into(),
        ],
    );
}
