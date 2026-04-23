use anyhow::Result;
use app::data_loader::Tokenizer;
use app::model::{Config, TransformerLanguageModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{distributions::WeightedIndex, prelude::Distribution};
use std::collections::HashMap;
use std::fs::File;

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("Loading tokenizer...");
    let f = File::open("tokenizer.bin")?;
    let (chars, stoi, itos, vocab_size): (
        Vec<char>,
        HashMap<char, usize>,
        HashMap<usize, char>,
        usize,
    ) = bincode::deserialize_from(f)?;
    let tokenizer = Tokenizer {
        chars,
        stoi,
        itos,
        vocab_size,
    };

    println!("Loading model...");
    let mut varmap = VarMap::new();
    varmap.load("model.safetensors")?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = Config { vocab_size: tokenizer.vocab_size, ..Default::default() };
    let model = TransformerLanguageModel::load(vb, config.clone())?;

    println!("Generating report based on database records...\n");
    let prompt = "dummy";
    let mut context = tokenizer.encode(prompt);

    if context.is_empty() {
        context.push(0);
    }

    let mut generated_ids = context.clone();
    let max_new_tokens = 50;
    let mut rng = rand::thread_rng();

    for _ in 0..max_new_tokens {
        // Crop context to block_size
        let start_idx = if generated_ids.len() > config.block_size {
            generated_ids.len() - config.block_size
        } else {
            0
        };
        let cond = &generated_ids[start_idx..];

        let idx_t = Tensor::from_vec(
            cond.iter().map(|&x| x as u32).collect(),
            (1, cond.len()),
            &device,
        )?;
        let logits = model.forward(&idx_t)?;

        let (_b, t, _c) = logits.dims3()?;
        let logits = logits.narrow(1, t - 1, 1)?.squeeze(1)?; // (1, vocab_size)
        let logits_slice = logits.to_vec2::<f32>()?;

        let probs: Vec<f32> = logits_slice[0].iter().map(|x| x.exp()).collect();
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|x| x / sum).collect();

        let dist = WeightedIndex::new(&probs).unwrap();
        let next_id = dist.sample(&mut rng);

        generated_ids.push(next_id);
    }

    let generated_text = tokenizer.decode(&generated_ids);
    println!("--- Automated Report ---");
    println!("{}", generated_text);
    println!("------------------------");

    Ok(())
}
