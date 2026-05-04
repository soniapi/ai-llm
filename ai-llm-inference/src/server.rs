use axum::{routing::get, Router, extract::State, Json};
use serde::Serialize;
use std::env;
use std::sync::Arc;
use tokio::net::TcpListener;
use tonic::transport::Channel;

use crate::ai_infra::context_service_client::ContextServiceClient;
use crate::ai_infra::HypothesisContextRequest;
use crate::tokenizer::Tokenizer;
use crate::LLM;
use tonic::Request;
use std::str::FromStr;

use std::fmt::Write;
use std::future::Future;
use std::pin::Pin;

type TokenProviderFn = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> + Send + Sync>;

pub struct AppState {
    pub llm: LLM,
    pub tokenizer: Tokenizer,
    pub grpc_channel: Channel,
    pub token_provider: TokenProviderFn,
}

pub fn create_app(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route("/generate", get(generate_hypothesis))
        .with_state(app_state)
}

pub async fn start_server(app_state: AppState) -> Result<(), Box<dyn std::error::Error>> {
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    println!("Listening on {}", addr);

    let shared_state = Arc::new(app_state);
    let app = create_app(shared_state);

    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn get_identity_token() -> Result<String, String> {
    let audience = "https://server-807069273288.us-central1.run.app";

    let provider = gcp_auth::provider().await
        .map_err(|e| format!("Failed to initialize GCP auth provider: {}", e))?;

    let token = provider.id_token(audience).await
        .map_err(|e| format!("Failed to obtain GCP identity token: {}", e))?;

    Ok(token.as_str().to_string())
}

#[derive(Serialize)]
pub struct HypothesisResponse {
    pub hypothesis: String,
}

async fn generate_hypothesis(State(state): State<Arc<AppState>>) -> Result<Json<HypothesisResponse>, String> {
    let token = (state.token_provider)().await?;
    let bearer_token = format!("Bearer {}", token);

    let channel = state.grpc_channel.clone();

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
    let response = client.get_hypothesis_context(request).await.map_err(|e| e.to_string())?.into_inner();

    let mut problem_statement = String::from("Based on the database schema payload:\n");
    for col in response.schema {
        let _ = write!(problem_statement, "Column: {}, Type: {}, Partition Key: {}\n", col.column_name, col.data_type, col.is_partition_key);
    }
    for stat in response.stats {
        let _ = write!(problem_statement, "Stat: {}, Min: {}, Max: {}, Avg: {}, Rows: {}\n", stat.column_name, stat.min_value, stat.max_value, stat.average_value, stat.total_rows);
    }

    let generated_hypothesis = state.llm.generate(&problem_statement, &state.tokenizer, 20);

    Ok(Json(HypothesisResponse { hypothesis: generated_hypothesis }))
}
