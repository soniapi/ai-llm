mod tensor;
mod transformer;
mod tokenizer;

use tensor::Tensor;
use transformer::{TransformerBlock, Linear};
use tokenizer::Tokenizer;

pub struct LLM {
    pub token_embedding: Linear,
    pub blocks: Vec<TransformerBlock>,
    pub lm_head: Linear,
    pub d_model: usize,
}

impl LLM {
    pub fn new(vocab_size: usize, d_model: usize, num_layers: usize) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(d_model));
        }

        Self {
            token_embedding: Linear::new(vocab_size, d_model),
            blocks,
            lm_head: Linear::new(d_model, vocab_size),
            d_model,
        }
    }

    pub fn forward(&self, tokens: &[usize], vocab_size: usize) -> Tensor {
        let seq_len = tokens.len();

        // One-hot encode tokens and pass to embedding
        let mut one_hot = Tensor::zeros(vec![seq_len, vocab_size]);
        for (i, &token) in tokens.iter().enumerate() {
            one_hot.data[i * vocab_size + token] = 1.0;
        }

        let mut x = self.token_embedding.forward(&one_hot);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        self.lm_head.forward(&x)
    }

    pub fn generate(&self, prompt: &str, tokenizer: &Tokenizer, max_len: usize) -> String {
        let mut tokens = tokenizer.encode(prompt);
        let vocab_size = tokenizer.vocab_size();

        for _ in 0..max_len {
            let logits = self.forward(&tokens, vocab_size);

            // Get last token's logits
            let last_dim = *logits.shape.last().unwrap();
            let start_idx = (logits.shape[0] - 1) * last_dim;
            let last_logits = &logits.data[start_idx..start_idx + last_dim];

            // Greedy decoding (argmax)
            let mut max_val = std::f32::NEG_INFINITY;
            let mut next_token = 0;
            for (i, &val) in last_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    next_token = i;
                }
            }

            tokens.push(next_token);
        }

        tokenizer.decode(&tokens)
    }
}

fn main() {
    println!("Initializing Dynamic Hypothesis Generator LLM from scratch...");

    let tokenizer = Tokenizer::new();
    let vocab_size = tokenizer.vocab_size();
    let d_model = 64; // Small embedding size
    let num_layers = 2; // Tiny transformer

    let llm = LLM::new(vocab_size, d_model, num_layers);

    let problem_statement = "The system is experiencing high latency during peak hours.";
    println!("\nProblem Statement: {}", problem_statement);

    // In a real model, weights would be loaded/trained.
    // Here we generate pseudo-randomly based on dummy weights.
    let generated_hypothesis = llm.generate(problem_statement, &tokenizer, 20);

    println!("\nGenerated Hypothesis: {}", generated_hypothesis);
}
