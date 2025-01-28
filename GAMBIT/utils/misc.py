
class CustomTrainer(Trainer):
    def __init__(self, real_metrics, word2vec_model, tfidf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_metrics = real_metrics
        self.word2vec_model = word2vec_model
        self.tfidf = tfidf

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Decode logits to text sentences
        generated_ids = torch.argmax(logits, dim=-1)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Calculate reward-based loss using generated sentences
        reward = sum(
            calculate_vector_reward(text, self.real_metrics, self.word2vec_model, self.tfidf)
            for text in generated_text
        ) / len(generated_text)  # Average reward for the batch
        
        # Use negative reward as loss (since we want to maximize reward)
        loss = -reward
        
        return (loss, outputs) if return_outputs else loss


def finetune_flan_t5_with_lora(model, tokenizer,real_questions, dataset_path, output_name, num_epochs=1, word2vec_model=None, tfidf=None):
    """Fine-tune Flan-T5 model using LORA with a custom loss function"""
    # Load the Flan-T5 model and tokenizer

    real_metrics = calculate_distribution_metrics(real_questions, word2vec_model, tfidf)


    flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    # Define the LORA configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    # Apply LORA to the Flan-T5 model
    lora_model = get_peft_model(flan_t5_model, lora_config)
    # Load the dataset
    dataset = load_dataset('csv', data_files=dataset_path)['train']
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{output_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Create an instance of CustomTrainer
    trainer = CustomTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=flan_t5_tokenizer,
        real_metrics=real_metrics,
        word2vec_model=word2vec_model,
        tfidf=tfidf,
    )

    # Fine-tune the Flan-T5 model using LORA with custom loss
    trainer.train()

    # Save the fine-tuned model
    lora_model.save_pretrained(f"./models/{output_name}")
    return lora_model


def finetune_question_generator_with_vector_reward(
    model,
    tokenizer,
    real_questions,
    word2vec_model,
    tfidf,
    num_epochs=1,
    batch_size=10
):
    """
    Finetune question generator using vector-based rewards
    """
    # real_questions_val = generate_question(model, tokenizer, "Ask an interesting question:")
    # print("this is a test: --------------" , real_questions_val)
    # Setup embedding model and calculate real distribution metrics
    real_metrics = calculate_distribution_metrics(real_questions, word2vec_model, tfidf)

    # Use the passed in model directly
    ppo_config = PPOConfig(
        batch_size=batch_size,
        learning_rate=1e-7,
        mini_batch_size=batch_size // 2,
        steps=25
    )
    # real_questions_val = generate_question(model, tokenizer, "Ask an interesting question:")
    # print("this is a test: --------------" , real_questions_val)
    # Initialize PPO trainer with the passed in model
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer
    )
    # real_questions_val = generate_question(model, tokenizer, "Ask an interesting question:")
    # print("this is a test: --------------" , real_questions_val)
    print("Starting vector-based RLHF finetuning...")

    for epoch in range(num_epochs):
        batch_queries = []
        batch_responses = []
        batch_rewards = []
        # real_questions_val = generate_question(model, tokenizer, "Ask an interesting question:")
        # print("this is a test: --------------" , real_questions_val)
        # Generate questions and calculate rewards
        for _ in range(batch_size):
            # Generate question
            prompt = "Ask an interesting question:"
            question = generate_question(model, tokenizer, prompt)

            # Calculate reward using vector-based metrics
            reward = calculate_vector_reward(
                question,
                real_metrics,
                word2vec_model,
                tfidf
            )

            # Prepare inputs
            query_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            response_inputs = tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            batch_queries.append(query_inputs["input_ids"])
            batch_responses.append(response_inputs["input_ids"])
            batch_rewards.append(torch.tensor(reward).to(device))

        try:
            # Perform PPO update on the passed in model
            stats = ppo_trainer.step(
                batch_queries,
                batch_responses,
                batch_rewards
            )

            avg_reward = sum(batch_rewards) / len(batch_rewards)
            print(f"Epoch {epoch + 1}, Batch completed - Average reward: {avg_reward:.4f}")

        except Exception as e:
            print(f"Error during PPO step: {e}")
            continue

    return model

