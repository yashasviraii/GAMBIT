from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

import os
import torch
from trl.models import AutoModelForSeq2SeqLMWithValueHead



def setup_rlhf_components(QUESTION_MODEL_NAME, device = 'cuda'):
    """Setup components needed for RLHF training"""
    from trl import PPOTrainer, PPOConfig
    from trl.models import AutoModelForSeq2SeqLMWithValueHead

    # Convert Flan-T5 model to include value head for PPO
    policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        QUESTION_MODEL_NAME
    ).to(device)

    # Define PPO configuration
    ppo_config = PPOConfig(
        batch_size=8,
        learning_rate=1e-7,
        mini_batch_size=4,
        steps=25
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        model=policy_model,
        config=ppo_config,
        tokenizer=T5Tokenizer.from_pretrained(QUESTION_MODEL_NAME)
    )

    return policy_model, ppo_trainer


def setup_detection_model(DETECTION_MODEL_NAME , device = 'cuda'):
    """Setup AI text detection model"""
    local_model_dir = "./MAGE/model"

    # Create directory if it doesn't exist
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    # Check if model is already downloaded locally
    if (os.path.exists(os.path.join(local_model_dir, "pytorch_model.bin")) and
        os.path.exists(os.path.join(local_model_dir, "tokenizer_config.json"))):
        print("Loading detection model and tokenizer from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_dir).to(device)
    else:
        # Download model and tokenizer and save them locally
        print("Downloading detection model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(DETECTION_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(DETECTION_MODEL_NAME).to(device)

        # Save the downloaded model and tokenizer locally
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)
        print(f"Detection model and tokenizer saved to {local_model_dir}")

    return model, tokenizer




def setup_quantized_model(MODEL_NAME):
    """Setup quantized model with similar configuration to original script"""
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer




def load_quantized_model(model_path , config_path):
    from peft import PeftModel, PeftConfig
    """
    Setup a quantized LLaMA model with LoRA adapters.

    Args:
        adapter_config_path (str): Path to the LoRA adapter configuration (adapter_config.json).
        model_weights_path (str): Path to the LoRA adapter model weights (adapter_model.bin).

    Returns:
        model: The quantized LLaMA model with LoRA applied.
        tokenizer: Tokenizer for the base model.
    """
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load the base model and tokenizer
    base_model_name = "openlm-research/open_llama_3b"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    import json
    # Load LoRA adapter
    with open(config_path, "r") as f:
        adapter_config_dict = json.load(f)

    # Convert dictionary into PeftConfig object
    adapter_config = PeftConfig(**adapter_config_dict)

    adapter_config.base_model_name_or_path = base_model_name  # Ensure base model name is set
    model = PeftModel(base_model, adapter_config)

    # Load LoRA weights
    lora_weights = torch.load(model_path, map_location="cpu")
    model.load_adapter(lora_weights)

    # Final model configuration
    model.eval()
    model.config.use_cache = False

    return model, tokenizer




def load_model_and_tokenizer(model_path: str, use_4bit: bool = True, 
                             bnb_4bit_compute_dtype: str = "bfloat16", 
                             bnb_4bit_quant_type: str = "nf4", use_nested_quant: bool = False):
    """
    Loads a quantized model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the saved model folder.
        use_4bit (bool): Whether to load the model with 4-bit quantization.
        bnb_4bit_compute_dtype (str): The data type to use for 4-bit computation (e.g., "bfloat16", "float16").
        bnb_4bit_quant_type (str): The type of 4-bit quantization (e.g., "nf4").
        use_nested_quant (bool): Whether to use double quantization.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    
    # Set up 4-bit quantization configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)  # Resolve torch dtype
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load the model with the quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically assign to GPU/CPU as appropriate
    )
    model.config.use_cache = False  # Prevent caching for PEFT compatibility

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS
    tokenizer.padding_side = "right"  # Padding on the right side

    # Return the loaded model and tokenizer
    return model, tokenizer



def setup_question_model(QUESTION_MODEL_NAME , device = 'cuda'):
    """Setup Flan-T5 for question generation"""
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(QUESTION_MODEL_NAME).to(device)
    tokenizer = T5Tokenizer.from_pretrained(QUESTION_MODEL_NAME)



    return model, tokenizer

