use ai_llm_training::run_training_loop;

fn main() {
    println!("Example: Initializing the AI-LLM Modulith Training Job...");

    // This runs the self-supervised training process by connecting to the database
    // and processing raw data into sequences.
    run_training_loop();

    println!("Example: Training Job finished successfully.");
}
