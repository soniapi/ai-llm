use ai_llm_inference::ai_infra::context_service_server::{ContextService, ContextServiceServer};
use ai_llm_inference::ai_infra::{ColumnDefinition, ColumnStatistics, HypothesisContextRequest, HypothesisContextResponse};
use ai_llm_inference::server::{create_app, AppState};
use ai_llm_inference::{LLM, tokenizer::Tokenizer};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt; // for `collect`
use std::sync::Arc;
use tokio::net::TcpListener;
use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
use tower::ServiceExt;

#[derive(Default)]
struct MockContextService {}

#[tonic::async_trait]
impl ContextService for MockContextService {
    async fn get_hypothesis_context(
        &self,
        request: TonicRequest<HypothesisContextRequest>,
    ) -> Result<TonicResponse<HypothesisContextResponse>, Status> {
        let _req = request.into_inner();

        let response = HypothesisContextResponse {
            schema: vec![
                ColumnDefinition {
                    column_name: "id".to_string(),
                    data_type: "int".to_string(),
                    is_partition_key: true,
                },
                ColumnDefinition {
                    column_name: "name".to_string(),
                    data_type: "varchar".to_string(),
                    is_partition_key: false,
                },
            ],
            stats: vec![
                ColumnStatistics {
                    column_name: "id".to_string(),
                    min_value: 1.0,
                    max_value: 100.0,
                    average_value: 50.5,
                    total_rows: 100,
                },
            ],
            active_partitions: vec![],
        };

        Ok(TonicResponse::new(response))
    }
}

async fn start_mock_server() -> tonic::transport::Channel {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let port = addr.port();
    drop(listener);

    let addr_str = format!("127.0.0.1:{}", port);
    let socket_addr = addr_str.parse().unwrap();

    let service = ContextServiceServer::new(MockContextService::default());

    tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve(socket_addr)
            .await
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    tonic::transport::Channel::from_shared(format!("http://{}", addr_str))
        .unwrap()
        .connect()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_generate_hypothesis_integration() {
    let grpc_channel = start_mock_server().await;

    let tokenizer = Tokenizer::new();
    let vocab_size = tokenizer.vocab_size();
    let llm = LLM::new(vocab_size, 64, 2);

    let app_state = AppState {
        llm,
        tokenizer,
        grpc_channel,
        token_provider: Arc::new(|| Box::pin(async { Ok("dummy-test-token".to_string()) })),
    };

    let app = create_app(Arc::new(app_state));

    let response = app
        .oneshot(Request::builder().uri("/generate").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    assert!(body_str.contains("\"hypothesis\":"));
}
