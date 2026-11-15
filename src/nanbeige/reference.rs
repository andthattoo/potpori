//! The same two models in plain Rust, one token at a time, with f64 accumulators: what the
//! kernels are checked against. Written from the published architectures, not from the GPU
//! code, and far too slow for anything but generated models of a few hundred channels.

use super::config::{rope_table, Config, DraftConfig, HEAD_DIM};
use super::weights::Checkpoint;

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &w)| (v as f64 * scale * w as f64) as f32)
        .collect()
}

/// `w` is `[n_out, n_in]` row-major.
fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks(x.len())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotate each head of `x` (`[heads, HEAD_DIM]` flat) for position `pos`.
fn rope(x: &mut [f32], table: &[f32], pos: usize) {
    let half = HEAD_DIM / 2;
    let row = &table[pos * HEAD_DIM..(pos + 1) * HEAD_DIM];
    for head in x.chunks_mut(HEAD_DIM) {
        for i in 0..half {
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * row[i] - b * row[half + i];
            head[i + half] = b * row[i] + a * row[half + i];
        }
    }
}

/// One query token (`[heads, HEAD_DIM]`) over cached keys and values (`[kv_heads, HEAD_DIM]`
/// per position).
fn attend(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    heads: usize,
    kv_heads: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; heads * HEAD_DIM];
    for h in 0..heads {
        let kv = h / (heads / kv_heads) * HEAD_DIM;
        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let scores: Vec<f64> = keys
            .iter()
            .map(|k| {
                qh.iter()
                    .zip(&k[kv..kv + HEAD_DIM])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>()
                    / (HEAD_DIM as f64).sqrt()
            })
            .collect();
        let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - best).exp()).collect();
        let total: f64 = weights.iter().sum();
        for d in 0..HEAD_DIM {
            let acc: f64 = weights
                .iter()
                .zip(values)
                .map(|(w, v)| w * v[kv + d] as f64)
                .sum();
            out[h * HEAD_DIM + d] = (acc / total) as f32;
        }
    }
    out
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

/// The whole target model.
pub struct RefTarget<'a> {
    cfg: Config,
    ck: &'a Checkpoint,
    table: Vec<f32>,
    /// `[layer][position]`
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

impl<'a> RefTarget<'a> {
    pub fn new(cfg: Config, ck: &'a Checkpoint, max_seq: usize) -> Self {
        Self {
            cfg,
            ck,
            table: rope_table(cfg.rope_theta, max_seq),
            keys: vec![Vec::new(); cfg.layers],
            values: vec![Vec::new(); cfg.layers],
        }
    }

    fn w(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.ck.dense(name, shape).unwrap()
    }

    /// Feed `token` at `pos` (anything cached from `pos` on is dropped). Returns the logits
    /// and the residual stream after every layer.
    pub fn feed(&mut self, pos: usize, token: i32) -> (Vec<f32>, Vec<Vec<f32>>) {
        let c = self.cfg;
        let (hidden, inter) = (c.hidden, c.intermediate);
        let (q_dim, kv_dim) = (c.q_dim(), c.kv_dim());
        let embed = self.w("model.embed_tokens.weight", &[c.vocab, hidden]);
        let mut h = embed[token as usize * hidden..(token as usize + 1) * hidden].to_vec();
        let mut after = Vec::new();
        for l in 0..c.layers {
            let p = |name: &str| format!("model.layers.{l}.{name}.weight");
            let xn = rms(
                &h,
                &self.w(&p("input_layernorm"), &[hidden]),
                c.rms_norm_eps,
            );
            let mut q = matvec(&self.w(&p("self_attn.q_proj"), &[q_dim, hidden]), &xn);
            let mut k = matvec(&self.w(&p("self_attn.k_proj"), &[kv_dim, hidden]), &xn);
            let v = matvec(&self.w(&p("self_attn.v_proj"), &[kv_dim, hidden]), &xn);
            rope(&mut q, &self.table, pos);
            rope(&mut k, &self.table, pos);
            self.keys[l].truncate(pos);
            self.values[l].truncate(pos);
            self.keys[l].push(k);
            self.values[l].push(v);
            let a = attend(&q, &self.keys[l], &self.values[l], c.n_heads, c.n_kv_heads);
            add(
                &mut h,
                &matvec(&self.w(&p("self_attn.o_proj"), &[hidden, q_dim]), &a),
            );
            let xn = rms(
                &h,
                &self.w(&p("post_attention_layernorm"), &[hidden]),
                c.rms_norm_eps,
            );
            let g = matvec(&self.w(&p("mlp.gate_proj"), &[inter, hidden]), &xn);
            let u = matvec(&self.w(&p("mlp.up_proj"), &[inter, hidden]), &xn);
            let m: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
            add(
                &mut h,
                &matvec(&self.w(&p("mlp.down_proj"), &[hidden, inter]), &m),
            );
            after.push(h.clone());
        }
        let xn = rms(&h, &self.w("model.norm.weight", &[hidden]), c.rms_norm_eps);
        let logits = matvec(&self.w("lm_head.weight", &[c.vocab, hidden]), &xn);
        (logits, after)
    }
}

/// The drafter's layer stack.
pub struct RefDraft<'a> {
    cfg: DraftConfig,
    ck: &'a Checkpoint,
    /// The target's embedding table.
    embed: Vec<f32>,
    table: Vec<f32>,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

impl<'a> RefDraft<'a> {
    pub fn new(cfg: DraftConfig, ck: &'a Checkpoint, embed: Vec<f32>, max_seq: usize) -> Self {
        Self {
            table: rope_table(cfg.rope_theta, max_seq),
            keys: vec![Vec::new(); cfg.layers],
            values: vec![Vec::new(); cfg.layers],
            cfg,
            ck,
            embed,
        }
    }

    fn w(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.ck.f32(name, shape).unwrap()
    }

    fn head_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        x.chunks(HEAD_DIM)
            .flat_map(|head| rms(head, weight, self.cfg.rms_norm_eps))
            .collect()
    }

    /// Target features for position `pos`: one row of `taps * hidden`.
    pub fn observe(&mut self, pos: usize, features: &[f32]) {
        let c = self.cfg.clone();
        let (hidden, kv_dim) = (c.hidden, c.kv_dim());
        let fc = self.w("fc.weight", &[hidden, c.taps.len() * hidden]);
        let ctx = rms(
            &matvec(&fc, features),
            &self.w("hidden_norm.weight", &[hidden]),
            c.rms_norm_eps,
        );
        for l in 0..c.layers {
            let p = |name: &str| format!("layers.{l}.self_attn.{name}.weight");
            let k = matvec(&self.w(&p("k_proj"), &[kv_dim, hidden]), &ctx);
            let mut k = self.head_norm(&k, &self.w(&p("k_norm"), &[HEAD_DIM]));
            rope(&mut k, &self.table, pos);
            let v = matvec(&self.w(&p("v_proj"), &[kv_dim, hidden]), &ctx);
            self.keys[l].truncate(pos);
            self.values[l].truncate(pos);
            self.keys[l].push(k);
            self.values[l].push(v);
        }
    }

    /// `y[p] = (base[side][0] + coef[p][side][0]) * x[p] + (base[side][1] + coef[p][side][1]) * x[p-1]`
    fn conv(&self, x: &[Vec<f32>], coef: &[Vec<f32>], base: &[f32], side: usize) -> Vec<Vec<f32>> {
        let (hidden, group) = (self.cfg.hidden, self.cfg.conv_group);
        let groups = hidden / group;
        (0..x.len())
            .map(|p| {
                (0..hidden)
                    .map(|ch| {
                        let g = ch / group;
                        let at = |tap: usize| {
                            base[(side * 2 + tap) * hidden + ch]
                                + coef[p][(side * 2 + tap) * groups + g]
                        };
                        let mut y = at(0) * x[p][ch];
                        if p > 0 {
                            y += at(1) * x[p - 1][ch];
                        }
                        y
                    })
                    .collect()
            })
            .collect()
    }

    /// The final hidden states of the block `[anchor, mask, mask, …]` at `start`, having
    /// observed every position before it.
    pub fn block(&self, anchor: i32, start: usize) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let (hidden, inter, block) = (c.hidden, c.intermediate, c.block);
        let (q_dim, kv_dim) = (c.q_dim(), c.kv_dim());
        let coef_n = 4 * c.conv_groups();
        let row = |t: i32| self.embed[t as usize * hidden..(t as usize + 1) * hidden].to_vec();
        let mut h: Vec<Vec<f32>> = (0..block)
            .map(|i| row(if i == 0 { anchor } else { c.mask_token }))
            .collect();
        let each = |xs: &[Vec<f32>], f: &dyn Fn(&[f32]) -> Vec<f32>| -> Vec<Vec<f32>> {
            xs.iter().map(|x| f(x)).collect()
        };
        for l in 0..c.layers {
            let p = |name: &str| format!("layers.{l}.{name}");
            let w = |name: &str, shape: &[usize]| self.w(&p(name), shape);

            let ln1 = w("input_layernorm.weight", &[hidden]);
            let xn = each(&h, &|x| rms(x, &ln1, c.rms_norm_eps));
            let proj = w("attention_conv.kernel_projection.weight", &[coef_n, hidden]);
            let base = w("attention_conv.base_kernel", &[2, 2, hidden]);
            let coef = each(&xn, &|x| matvec(&proj, x));
            let xc = self.conv(&xn, &coef, &base, 0);

            let (wq, wk, wv) = (
                w("self_attn.q_proj.weight", &[q_dim, hidden]),
                w("self_attn.k_proj.weight", &[kv_dim, hidden]),
                w("self_attn.v_proj.weight", &[kv_dim, hidden]),
            );
            let (qn, kn) = (
                w("self_attn.q_norm.weight", &[HEAD_DIM]),
                w("self_attn.k_norm.weight", &[HEAD_DIM]),
            );
            let mut keys = self.keys[l][..start].to_vec();
            let mut values = self.values[l][..start].to_vec();
            let mut queries = Vec::new();
            for (i, x) in xc.iter().enumerate() {
                let mut q = self.head_norm(&matvec(&wq, x), &qn);
                let mut k = self.head_norm(&matvec(&wk, x), &kn);
                rope(&mut q, &self.table, start + i);
                rope(&mut k, &self.table, start + i);
                queries.push(q);
                keys.push(k);
                values.push(matvec(&wv, x));
            }
            let wo = w("self_attn.o_proj.weight", &[hidden, q_dim]);
            let y = each(&queries, &|q| {
                matvec(&wo, &attend(q, &keys, &values, c.n_heads, c.n_kv_heads))
            });
            for (h, y) in h.iter_mut().zip(self.conv(&y, &coef, &base, 1)) {
                add(h, &y);
            }

            let ln2 = w("post_attention_layernorm.weight", &[hidden]);
            let xn = each(&h, &|x| rms(x, &ln2, c.rms_norm_eps));
            let proj = w("mlp_conv.kernel_projection.weight", &[coef_n, hidden]);
            let base = w("mlp_conv.base_kernel", &[2, 2, hidden]);
            let coef = each(&xn, &|x| matvec(&proj, x));
            let xc = self.conv(&xn, &coef, &base, 0);
            let (wg, wu, wd) = (
                w("mlp.gate_proj.weight", &[inter, hidden]),
                w("mlp.up_proj.weight", &[inter, hidden]),
                w("mlp.down_proj.weight", &[hidden, inter]),
            );
            let y = each(&xc, &|x| {
                let (g, u) = (matvec(&wg, x), matvec(&wu, x));
                let m: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
                matvec(&wd, &m)
            });
            for (h, y) in h.iter_mut().zip(self.conv(&y, &coef, &base, 1)) {
                add(h, &y);
            }
        }
        let norm = self.w("norm.weight", &[hidden]);
        each(&h, &|x| rms(x, &norm, c.rms_norm_eps))
    }

    /// The two projections of the masked slots that feed the selector: logits through the
    /// target's `lm_head` and the selector's hidden projection.
    pub fn slot_outputs(&self, hidden_rows: &[Vec<f32>], lm_head: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let proj = self.w(
            "candidate_selector.hidden_projection.weight",
            &[self.cfg.selector_rank, self.cfg.hidden],
        );
        let slots = &hidden_rows[1..];
        (
            slots.iter().flat_map(|h| matvec(lm_head, h)).collect(),
            slots.iter().flat_map(|h| matvec(&proj, h)).collect(),
        )
    }
}
