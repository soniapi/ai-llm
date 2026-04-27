use reqwest;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the base URL from the command line or environment, defaulting to the Cloud Run URL
    let base_url = env::args()
        .nth(1)
        .unwrap_or_else(|| "https://ai-llm-5u7ahgmduq-uc.a.run.app".to_string());

    let url = format!("{}/generate", base_url);

    println!("Calling ai-llm service at: {}", url);

    // Make a GET request to the /generate endpoint
    let response = reqwest::get(&url).await?;

    if response.status().is_success() {
        let hypothesis = response.text().await?;
        println!("\nGenerated Hypothesis:");
        println!("---------------------");
        println!("{}", hypothesis);
        println!("---------------------");
    } else {
        eprintln!("Error: Received status code {}", response.status());
        let error_body = response.text().await?;
        eprintln!("Response body: {}", error_body);
    }

    Ok(())
}
