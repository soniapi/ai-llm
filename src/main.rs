#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ai_hypo::run_main().await
}
