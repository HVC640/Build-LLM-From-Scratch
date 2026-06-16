
import os
from pathlib import Path
import time
import pandas as pd
from functools import partial

import torch
import tiktoken

from attention.causal_attention import CausalAttention
from attention.self_attention import SelfAttention_v1, SelfAttention_v2
from attention.multi_head_attention import MultiHeadAttention, MultiHeadAttentionWrapper
from datasets.instruction_dataset import InstructionDataset, download_and_load_file
from datasets.spam_dataset import download_and_unzip_spam_data, create_balanced_dataset, random_split
from datasets.spam_dataset import SpamDataset
from tokenizer.dataset import create_dataloader_v1
from tokenizer.tokenizer import GPTTokenizer


from torch.utils.data import DataLoader
from embeddings.token_embedding import TokenEmbedding
from embeddings.positional_embedding import PositionalEmbedding

from training.collate import custom_collate_draft_1, custom_collate_draft_2, custom_collate_fn
from training.loss import calc_accuracy_loader, calc_classifier_finetune_loss_loader, calc_instruction_loss_loader
from transformer.gpt_model import GPTModel
from training.generation import classify_review, generate, generate_text_simple
from training.trainer import train_classifier_simple, train_model_simple

from datasets.gpt_download3 import download_and_load_gpt2, load_weights_into_gpt
from utils.seed import format_input, text_to_token_ids, token_ids_to_text


def test_tiktoken():
    # Get the encoding for the model
    encoding = tiktoken.encoding_for_model("gpt-2")

    # Encode a string into tokens
    tokens = encoding.encode("Hello, world!")
    print(tokens)  # Returns a list of token integers
    token_texts = [encoding.decode([token]) for token in tokens]
    print(token_texts)  # Returns the text representation of each token

    # Decode tokens back into text
    text = encoding.decode(tokens)
    print(text)  # Returns "Hello, world!"

    # Get the exact token count
    token_count = len(tokens)
    print(f"Token count: {token_count}")

    tokens = encoding.encode(
        'def check_odd_even(num: int) -> str:    return "Even" if num % 2 == 0 else "Odd"')
    print(tokens)  # Returns a list of token integers
    token_texts = [encoding.decode([token]) for token in tokens]
    print(token_texts)  # Returns the text representation of each token

    text = encoding.decode(tokens)
    print(text)  # Returns "The quick brown fox jumps over the lazy dog."

    token_count = len(tokens)
    print(f"Token count: {token_count}")


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


def test_gpt_model():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Example batch of token indices
    # Batch size of 2, sequence length of 4
    batch = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 4))

    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = total_params - \
        sum(p.numel() for p in model.out_head.parameters())
    print(
        f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Assuming 4 bytes per parameter (float32)
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    tokenizer = GPTTokenizer(encoding_name="gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(
        0)  # Shape (1, sequence_length)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()  # Set the model to evaluation mode
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)


def test_training_loop():
    import time
    start_time = time.time()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPTTokenizer(encoding_name="gpt2")

    # Train/validation ratio
    text_data = ""
    with open("C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\the-verdict.txt", "r") as f:
        text_data = f.read()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def test_download_gpt2():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    model_size = "124M"
    models_dir = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\models"
    model_dir = os.path.join(models_dir, f"gpt2-{model_size}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPTTokenizer(encoding_name="gpt2")

    settings, params = download_and_load_gpt2(model_size, model_dir)

    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())

    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def test_finetune_classifier():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\sms_spam_collection.zip"
    extracted_path = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    train_csv_path = Path(
        "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\train.csv")
    validation_csv_path = Path(
        "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\validation.csv")
    test_csv_path = Path(
        "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\test.csv")

    finetuned_model_path = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\models\\finetuned_review_classifier.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t",
                     header=None, names=["Label", "Text"])
    print(df.head())
    print(df["Label"].value_counts())

    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    print(len(train_df))
    print(len(validation_df))
    print(len(test_df))

    tokenizer = GPTTokenizer(encoding_name="gpt2")

    train_df.to_csv(train_csv_path, index=None)
    validation_df.to_csv(validation_csv_path, index=None)
    test_df.to_csv(test_csv_path, index=None)

    train_dataset = SpamDataset(
        csv_file=train_csv_path,
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=validation_csv_path,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_csv_path,
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    print(train_dataset.max_length)
    print(val_dataset.max_length)
    print(test_dataset.max_length)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    models_dir = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\models\\gpt2-124M\\124M"
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=models_dir)

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    # shape: (batch_size, num_tokens)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    # shape: (batch_size, num_tokens, num_classes)
    print("Outputs dimensions:", outputs.shape)

    print("Last output token:", outputs[:, -1, :])

    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_classifier_finetune_loss_loader(
            train_loader, model, device, num_batches=5)
        val_loss = calc_classifier_finetune_loss_loader(
            val_loader, model, device, num_batches=5)
        test_loss = calc_classifier_finetune_loss_loader(
            test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    torch.save(model.state_dict(), finetuned_model_path)

    # model_state_dict = torch.load("review_classifier.pth")
    # model.load_state_dict(model_state_dict)


def test_finetune_instruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model_path = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\models\\finetuned_instruction_model.pth"
    file_path = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\assets\\instruction-data.json"
    models_dir = "C:\\Workspace\\projects\\Build-LLM-From-Scratch\\models\\gpt2-124M\\124M"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    print("Example entry:\n", data[50])

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    print(custom_collate_draft_1(batch))

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )
    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=models_dir
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    model.to(device)
    with torch.no_grad():
        train_loss = calc_instruction_loss_loader(
            train_loader, model, device, num_batches=5)
        val_loss = calc_instruction_loss_loader(
            val_loader, model, device, num_batches=5)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 1

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    for entry in test_data[:3]:
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    torch.save(model.state_dict(), finetuned_model_path)


if __name__ == "__main__":
    # test_tokenizer_and_dataloader()
    # test_attention_modules()
    # test_tiktoken()
    # test_gpt_model()
    # test_training_loop()
    # test_download_gpt2()
    # test_finetune_classifier()
    test_finetune_instruction()
