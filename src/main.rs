#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ai_llm::run_main().await
}
