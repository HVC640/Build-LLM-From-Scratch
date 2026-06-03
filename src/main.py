
import torch

from embeddings import token_embedding
from tokenizer.dataset import create_dataloader_v1

from embeddings.token_embedding import TokenEmbedding
from embeddings.positional_embedding import PositionalEmbedding

if __name__ == "__main__":
    # Example usage of the tokenizer and dataloader
    with open("C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\the-verdict.txt", "r") as f:
        verdict_text = f.read()

    batch_size = 8
    max_length = 4
    stride = max_length
    dataloader = create_dataloader_v1(
        verdict_text, batch_size=batch_size, max_length=max_length, stride=stride)
    print(f"shape of dataloader: {len(dataloader)} batches")

    for input_ids, target_ids in dataloader:
        print("Input IDs:", input_ids)
        print("Input IDs shape:", input_ids.shape)
        print("Target IDs:", target_ids)
        print("Target IDs shape:", target_ids.shape)
        break  # Just show the first batch for demonstration

    # Example usage of the token embedding
    vocab_size = 50257  # GPT-2 vocab size
    embedding_dim = 256  # Example embedding dimension
    token_embedding_layer = TokenEmbedding(vocab_size, embedding_dim)
    sample_token_ids = input_ids  # Take the input_ids from the dataloader as sample token ids
    input_embeddings = token_embedding_layer(sample_token_ids)
    print("Input Embeddings:", input_embeddings)
    print("Input Embeddings shape:", input_embeddings.shape)

    # Example usage of the positional embedding
    context_length = max_length  # Use the same max_length as context length for positional embedding
    positional_embedding_layer = PositionalEmbedding(
        context_length, embedding_dim)
    # Position indices for the sequence
    sample_positions = torch.arange((sample_token_ids).shape[1])  # Shape: (batch_size, sequence_length)
    pos_embeddings = positional_embedding_layer(sample_positions)
    print("Positional Embeddings:", pos_embeddings)
    print("Positional Embeddings shape:", pos_embeddings.shape)

    # Example usage of combining token and positional embeddings
    combined_embeddings = input_embeddings + pos_embeddings
    print("Combined Embeddings:", combined_embeddings)
    print("Combined Embeddings shape:", combined_embeddings.shape)
