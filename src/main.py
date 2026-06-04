
import torch

from attention.causal_attention import CausalAttention
from attention.self_attention import SelfAttention_v1, SelfAttention_v2
from attention.multi_head_attention import MultiHeadAttention, MultiHeadAttentionWrapper
from embeddings import token_embedding
from tokenizer.dataset import create_dataloader_v1

from embeddings.token_embedding import TokenEmbedding
from embeddings.positional_embedding import PositionalEmbedding


def test_tokenizer_and_dataloader():
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
    # Take the input_ids from the dataloader as sample token ids
    sample_token_ids = input_ids
    input_embeddings = token_embedding_layer(sample_token_ids)
    print("Input Embeddings:", input_embeddings)
    print("Input Embeddings shape:", input_embeddings.shape)

    # Example usage of the positional embedding
    # Use the same max_length as context length for positional embedding
    context_length = max_length
    positional_embedding_layer = PositionalEmbedding(
        context_length, embedding_dim)
    # Position indices for the sequence
    # Shape: (batch_size, sequence_length)
    sample_positions = torch.arange((sample_token_ids).shape[1])
    pos_embeddings = positional_embedding_layer(sample_positions)
    print("Positional Embeddings:", pos_embeddings)
    print("Positional Embeddings shape:", pos_embeddings.shape)

    # Example usage of combining token and positional embeddings
    combined_embeddings = input_embeddings + pos_embeddings
    print("Combined Embeddings:", combined_embeddings)
    print("Combined Embeddings shape:", combined_embeddings.shape)


def test_attention_modules():
    # Example usage of the attention modules
    batch_size = 1
    context_length = 6
    d_in = 4
    d_out = 6
    dropout_rate = 0.1  # Example dropout rate
    num_heads = 2  # Example number of attention heads

    # Create random input tensor simulating embedded tokens
    x = torch.rand(batch_size, context_length, d_in)

    # Test SelfAttention_v1
    self_attention_v1 = SelfAttention_v1(d_in, d_out)
    output_v1 = self_attention_v1(x)
    print("Output of SelfAttention_v1:", output_v1)
    print("Output shape of SelfAttention_v1:", output_v1.shape)

    # Test SelfAttention_v2
    self_attention_v2 = SelfAttention_v2(d_in, d_out)
    output_v2 = self_attention_v2(x)
    print("Output of SelfAttention_v2:", output_v2)
    print("Output shape of SelfAttention_v2:", output_v2.shape)

    # Test CausalAttention
    causal_attention = CausalAttention(
        d_in, d_out, context_length, dropout_rate)
    output_causal = causal_attention(x)
    print("Output of CausalAttention:", output_causal)
    print("Output shape of CausalAttention:", output_causal.shape)

    # Test MultiHeadAttentionWrapper
    multi_head_attention_wrapper = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, dropout_rate, num_heads)
    output_multi_head_wrapper = multi_head_attention_wrapper(x)
    print("Output of MultiHeadAttentionWrapper:", output_multi_head_wrapper)
    print("Output shape of MultiHeadAttentionWrapper:",
          output_multi_head_wrapper.shape)

    # Test MultiHeadAttention
    multi_head_attention = MultiHeadAttention(
        d_in, d_out, context_length, dropout_rate, num_heads)
    output_multi_head = multi_head_attention(x)
    print("Output of MultiHeadAttention:", output_multi_head)
    print("Output shape of MultiHeadAttention:", output_multi_head.shape)


if __name__ == "__main__":
    test_attention_modules()
