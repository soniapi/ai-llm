mod tensor;
mod transformer;
mod tokenizer;

use tensor::Tensor;
use transformer::{TransformerBlock, Linear};
use tokenizer::Tokenizer;

pub mod ai_infra {
    tonic::include_proto!("ai_infra");
}

use ai_infra::context_service_client::ContextServiceClient;
use ai_infra::HypothesisContextRequest;
use gcp_auth::TokenProvider;
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::Request;
use std::str::FromStr;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Dynamic Hypothesis Generator LLM from scratch...");

    let tokenizer = Tokenizer::new();
    let vocab_size = tokenizer.vocab_size();
    let d_model = 64; // Small embedding size
    let num_layers = 2; // Tiny transformer

    let llm = LLM::new(vocab_size, d_model, num_layers);

    // Authenticate with GCP using credentials from environment
    let authentication_manager = gcp_auth::provider().await.map_err(|e| format!("Failed to get GCP credentials: {}", e))?;
    let token = authentication_manager.token(&["https://www.googleapis.com/auth/cloud-platform"]).await.map_err(|e| format!("Failed to fetch token: {}", e))?;
    let bearer_token = format!("Bearer {}", token.as_str());

    // Setup gRPC client pointing to the Cloud Run service URL
    let tls_config = ClientTlsConfig::new().with_native_roots();
    let channel = Channel::from_static("https://server-807069273288.us-central1.run.app")
        .tls_config(tls_config)?
        .connect()
        .await?;

    let mut client = ContextServiceClient::with_interceptor(channel, move |mut req: Request<()>| {
        req.metadata_mut().insert(
            "authorization",
            tonic::metadata::MetadataValue::from_str(&bearer_token).unwrap(),
        );
        Ok(req)
    });

    let request = tonic::Request::new(HypothesisContextRequest {
        target_table: "my_table".into(),
        since_timestamp: "2024-01-01T00:00:00Z".into(),
    });

    println!("Requesting HypothesisContext from gRPC API...");
    let response = client.get_hypothesis_context(request).await?.into_inner();

    // Construct problem statement from the DB schema response
    let mut problem_statement = String::from("Based on the database schema payload:\n");
    for col in response.schema {
        problem_statement.push_str(&format!("Column: {}, Type: {}, Partition Key: {}\n", col.column_name, col.data_type, col.is_partition_key));
    }
    for stat in response.stats {
        problem_statement.push_str(&format!("Stat: {}, Min: {}, Max: {}, Avg: {}, Rows: {}\n", stat.column_name, stat.min_value, stat.max_value, stat.average_value, stat.total_rows));
    }

    println!("\nProblem Statement: {}", problem_statement);

    // In a real model, weights would be loaded/trained.
    // Here we generate pseudo-randomly based on dummy weights.
    let generated_hypothesis = llm.generate(&problem_statement, &tokenizer, 20);

    println!("\nGenerated Hypothesis: {}", generated_hypothesis);

    Ok(())
}
