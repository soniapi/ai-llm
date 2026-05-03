use ai_llm_training::run_training_loop;

fn main() {
    println!("Initializing the AI-LLM Modulith Training Job...");

    // Start the continuous self-supervised training process.
    // In a production environment, this would run continuously as a background
    // worker, or process chunks of data via a cron schedule.
    run_training_loop();

    println!("Training Job finished successfully.");
}
