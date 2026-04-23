#[cfg(test)]
mod tests {
    use super::*;
    use app::data_loader::Tokenizer;

    #[test]
    fn test_tokenizer() {
        let text = "hello world";
        let tokenizer = Tokenizer::new(text);

        let encoded = tokenizer.encode("hello");
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }
}
