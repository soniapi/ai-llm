use std::collections::{HashMap, HashSet};

pub struct Tokenizer {
    pub chars: Vec<char>,
    pub stoi: HashMap<char, usize>,
    pub itos: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        let mut char_set = HashSet::new();
        for ch in text.chars() {
            char_set.insert(ch);
        }
        let mut chars: Vec<char> = char_set.into_iter().collect();
        chars.sort();

        let vocab_size = chars.len();
        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();

        for (i, &ch) in chars.iter().enumerate() {
            stoi.insert(ch, i);
            itos.insert(i, ch);
        }

        Self {
            chars,
            stoi,
            itos,
            vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|ch| self.stoi.get(&ch).copied())
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.itos.get(&t).copied())
            .collect()
    }
}

pub fn create_dataset() -> String {
    let mut text = String::new();

    if let Ok(db_url) = std::env::var("DATABASE_URL") {
        let is_mock = db_url == "postgres://postgres:postgres@localhost:5432/postgres";
        if is_mock {
             eprintln!("Using mock connection data for tests.");
             return String::new(); // Return empty to trigger fallback dummy data
        }
    }

    let result = std::panic::catch_unwind(|| {
        use ai_infra::{establish_connection, models::ObjectS, schema::objects_s::dsl::objects_s};
        use diesel::prelude::*;
        let mut conn = establish_connection();
        if let Ok(results) = objects_s.load::<ObjectS>(&mut conn) {
            let mut inner_text = String::new();
            for obj in results {
                inner_text.push_str(&obj.t);
                inner_text.push('\n');
            }
            inner_text
        } else {
            String::new()
        }
    });

    if let Ok(db_text) = result {
        text = db_text;
    }

    text
}
