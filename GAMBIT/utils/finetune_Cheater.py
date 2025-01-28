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
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import torch

def just_fine_tune_model(model, tokenizer, dataset_path, output_name):
    """Fine-tune the model using the original script's approach"""
    # Load dataset
    dataset = load_dataset('csv', data_files=dataset_path)['train']

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=f"./results/{output_name}",
        num_train_epochs=1.5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2.5e-4,
        weight_decay=0.0003,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # Train
    trainer.train()

    # Save model
    # device_map = {"": 0}
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path= "openlm-research/open_llama_3b",
    #     # model_name = "openlm-research/open_llama_3b",
    #     low_cpu_mem_usage=True,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    #     device_map=device_map,
    # )
    # model = PeftModel.from_pretrained(base_model, r"D:\NLP_test\models\model2_round_13")
    # model = model.merge_and_unload()

    # # Reload tokenizer to save it
    # tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b", trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    # output_dir = "./model"  # specify the directory where you want to save the model
    # model.save_pretrained(output_dir)

    # # Save the tokenizer
    # tokenizer.save_pretrained(output_dir)

    # del base_model
    # del model
    trainer.model.save_pretrained(f"./models/{output_name}")
    return trainer.model


def fine_tune_model(model, tokenizer, dataset_path,MODEL_PATH, output_name):
    """Fine-tune the model using the original script's approach"""
    # Load dataset
    dataset = load_dataset('csv', data_files=dataset_path)['train']

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=f"./results/{output_name}",
        num_train_epochs=1.5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2.5e-4,
        weight_decay=0.0003,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # Train
    trainer.train()

    # Save model
    device_map = {"": 0}
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path= "openlm-research/open_llama_3b",
        # model_name = "openlm-research/open_llama_3b",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    output_dir = f"./model/{output_name}"  # specify the directory where you want to save the model
    model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    del base_model
    del model
    # trainer.model.save_pretrained(f"./models/{output_name}")
    return trainer.model

