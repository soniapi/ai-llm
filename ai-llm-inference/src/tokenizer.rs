use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<char, usize>,
    inverse_vocab: HashMap<usize, char>,
}

impl Tokenizer {
    pub fn new() -> Self {
        // Simple character level tokenizer
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:\n";
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();

        for (i, c) in chars.chars().enumerate() {
            vocab.insert(c, i);
            inverse_vocab.insert(i, c);
        }

        Self {
            vocab,
            inverse_vocab,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.vocab.get(&c).unwrap_or(&0)) // Map unknown to 0 for simplicity
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .map(|&t| *self.inverse_vocab.get(&t).unwrap_or(&'?')) // Map unknown to '?'
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
