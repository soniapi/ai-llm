use anyhow::Result;
use app::data_loader::{Tokenizer, create_dataset};
use app::model::{Config, TransformerLanguageModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::Rng;
use std::fs::File;

fn get_batch(
    data: &[usize],
    batch_size: usize,
    block_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let mut ix_x = Vec::with_capacity(batch_size * block_size);
    let mut ix_y = Vec::with_capacity(batch_size * block_size);

    for _ in 0..batch_size {
        let ix = rng.gen_range(0..data.len() - block_size);
        ix_x.extend_from_slice(&data[ix..ix + block_size]);
        ix_y.extend_from_slice(&data[ix + 1..ix + block_size + 1]);
    }

    let x = Tensor::from_vec(
        ix_x.into_iter().map(|v| v as u32).collect::<Vec<u32>>(),
        (batch_size, block_size),
        device,
    )?;
    let y = Tensor::from_vec(
        ix_y.into_iter().map(|v| v as u32).collect::<Vec<u32>>(),
        (batch_size, block_size),
        device,
    )?;

    Ok((x, y))
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Loading data from database...");
    let text = create_dataset();
    if text.is_empty() {
        println!("Database is empty or connection failed. Proceeding with dummy data for test.");
    }
    let text = if text.is_empty() {
        "this is a very long piece of dummy data for the transformer model so that the length is greater than the block size. ".repeat(10)
    } else {
        text
    };
    println!("Total text length: {}", text.len());

    let tokenizer = Tokenizer::new(&text);
    println!("Vocabulary size: {}", tokenizer.vocab_size);

    let data = tokenizer.encode(&text);
    let split_idx = (data.len() as f64 * 0.9) as usize;
    let train_data = &data[..split_idx];

    // Initialize the model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = Config { batch_size: 16, vocab_size: tokenizer.vocab_size, ..Default::default() };
    let model = TransformerLanguageModel::load(vb.clone(), config.clone())?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        },
    )?;

    println!("Starting training...");
    let max_iters = 20;
    let eval_interval = 10;

    for iter in 0..max_iters {
        if iter % eval_interval == 0 {
            let (xb, yb) = get_batch(train_data, config.batch_size, config.block_size, &device)?;
            let logits = model.forward(&xb)?;
            let (b, t, c) = logits.dims3()?;
            let logits = logits.reshape((b * t, c))?;
            let targets = yb.reshape((b * t,))?.to_dtype(DType::U32)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
            println!("Iter {}: loss {}", iter, loss.to_vec0::<f32>()?);
        }

        let (xb, yb) = get_batch(train_data, config.batch_size, config.block_size, &device)?;
        let logits = model.forward(&xb)?;

        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b * t, c))?;
        let targets = yb.reshape((b * t,))?.to_dtype(DType::U32)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;

        opt.backward_step(&loss)?;
    }

    println!("Saving model and tokenizer...");
    varmap.save("model.safetensors")?;

    // Save tokenizer details
    let vocab_data = (
        tokenizer.chars.clone(),
        tokenizer.stoi.clone(),
        tokenizer.itos.clone(),
        tokenizer.vocab_size,
    );
    let f = File::create("tokenizer.bin")?;
    bincode::serialize_into(f, &vocab_data)?;
    println!("Done!");

    Ok(())
}
