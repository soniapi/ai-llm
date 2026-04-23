use candle_core::{D, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub batch_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 65,
            block_size: 32,
            batch_size: 16,
            n_embd: 64,
            n_head: 4,
            n_layer: 4,
        }
    }
}

#[derive(Debug)]
pub struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    head_size: usize,
    tril: Tensor,
}

impl Head {
    pub fn load(
        vb: VarBuilder,
        head_size: usize,
        n_embd: usize,
        block_size: usize,
    ) -> Result<Self> {
        let key = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("key"))?;
        let query = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("query"))?;
        let value = candle_nn::linear_no_bias(n_embd, head_size, vb.pp("value"))?;

        let mut mask_data = vec![0u8; block_size * block_size];
        for r in 0..block_size {
            for c in 0..block_size {
                if c <= r {
                    mask_data[r * block_size + c] = 1;
                }
            }
        }
        let tril = Tensor::from_vec(mask_data, (block_size, block_size), vb.device())?;

        Ok(Self {
            key,
            query,
            value,
            head_size,
            tril,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t, _c) = x.dims3()?;
        let k = self.key.forward(x)?;
        let q = self.query.forward(x)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        let scale = (self.head_size as f64).sqrt();
        let wei = (q.matmul(&k_t)? / scale)?;

        let mask = self.tril.narrow(0, 0, t)?.narrow(1, 0, t)?;
        let mask = mask.broadcast_as(wei.shape())?;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, wei.device())?.broadcast_as(wei.shape())?;
        let wei = mask.where_cond(&wei, &neg_inf)?;

        let wei = candle_nn::ops::softmax(&wei, D::Minus1)?;

        let v = self.value.forward(x)?;
        let out = wei.matmul(&v)?;
        Ok(out)
    }
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
}

impl MultiHeadAttention {
    pub fn load(
        vb: VarBuilder,
        n_head: usize,
        head_size: usize,
        n_embd: usize,
        block_size: usize,
    ) -> Result<Self> {
        let mut heads = Vec::with_capacity(n_head);
        for i in 0..n_head {
            heads.push(Head::load(
                vb.pp(format!("heads.{}", i)),
                head_size,
                n_embd,
                block_size,
            )?);
        }
        let proj = candle_nn::linear(n_embd, n_embd, vb.pp("proj"))?;
        Ok(Self { heads, proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut head_outputs = Vec::with_capacity(self.heads.len());
        for h in &self.heads {
            head_outputs.push(h.forward(x)?);
        }
        let out = Tensor::cat(&head_outputs, 2)?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
pub struct FeedForward {
    net1: Linear,
    net2: Linear,
}

impl FeedForward {
    pub fn load(vb: VarBuilder, n_embd: usize) -> Result<Self> {
        let net1 = candle_nn::linear(n_embd, 4 * n_embd, vb.pp("net1"))?;
        let net2 = candle_nn::linear(4 * n_embd, n_embd, vb.pp("net2"))?;
        Ok(Self { net1, net2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.net1.forward(x)?;
        let x = x.relu()?;
        self.net2.forward(&x)
    }
}

#[derive(Debug)]
pub struct Block {
    sa: MultiHeadAttention,
    ffwd: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

impl Block {
    pub fn load(vb: VarBuilder, n_embd: usize, n_head: usize, block_size: usize) -> Result<Self> {
        let head_size = n_embd / n_head;
        let sa = MultiHeadAttention::load(vb.pp("sa"), n_head, head_size, n_embd, block_size)?;
        let ffwd = FeedForward::load(vb.pp("ffwd"), n_embd)?;
        let ln1 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln2"))?;
        Ok(Self { sa, ffwd, ln1, ln2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.sa.forward(&self.ln1.forward(x)?)?)?;
        let x = (&x + self.ffwd.forward(&self.ln2.forward(&x)?)?)?;
        Ok(x)
    }
}

#[derive(Debug)]
pub struct TransformerLanguageModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
    pub config: Config,
}

impl TransformerLanguageModel {
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let token_embedding_table = candle_nn::embedding(
            config.vocab_size,
            config.n_embd,
            vb.pp("token_embedding_table"),
        )?;
        let position_embedding_table = candle_nn::embedding(
            config.block_size,
            config.n_embd,
            vb.pp("position_embedding_table"),
        )?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block::load(
                vb.pp(format!("blocks.{}", i)),
                config.n_embd,
                config.n_head,
                config.block_size,
            )?);
        }

        let ln_f = candle_nn::layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = candle_nn::linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            token_embedding_table,
            position_embedding_table,
            blocks,
            ln_f,
            lm_head,
            config,
        })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;
        let tok_emb = self.token_embedding_table.forward(idx)?;

        let pos: Vec<u32> = (0..t as u32).collect();
        let pos = Tensor::new(pos.as_slice(), idx.device())?;
        let pos_emb = self.position_embedding_table.forward(&pos)?;

        let pos_emb = pos_emb.broadcast_as(tok_emb.shape())?;
        let mut x = (tok_emb + pos_emb)?;

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }
}
