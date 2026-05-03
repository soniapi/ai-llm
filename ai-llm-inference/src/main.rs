#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure rustls default crypto provider is initialized
    let _ = rustls::crypto::ring::default_provider().install_default();
    ai_llm_inference::run_main().await
}
