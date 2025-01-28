from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, TrainingArguments, Trainer

import time

from utils.finetune_pipeline import calculate_distribution_metrics , calculate_vector_reward , collect_initial_data

import torch


def finetune_detective_with_RLHF(model, tokenizer,real_questions, num_epochs=5, word2vec_model=None):
    """Fine-tune Flan-T5 model using LORA with a custom loss function"""
    # Load the Flan-T5 model and tokenizer
    rewards = []
    real_metrics = calculate_distribution_metrics(real_questions, word2vec_model)

    for i in real_questions:
         reward = calculate_vector_reward(
                i,
                real_metrics,
                word2vec_model
            )
         rewards.append(reward)


    ppo_config = PPOConfig(
    batch_size=8,
    learning_rate=1.41e-5,
    mini_batch_size=4,  # Adjusted to a smaller size for more frequent updates
    steps=2000,  # Number of PPO steps or epochs
    seed = int(time.time())
    )

    # Initialize PPO trainer with the model, configuration, and tokenizer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer
    )
    
    fixed_length = 100
    queries = []
    responses = []

    print("Fine-tuning Flan-T5 with PPO...")
    
    prompts = ["Ask an interesting question:"]*(len(real_questions))
    # Fine-tuning loop with PPO
    for epoch in range(len(prompts)):
        # Select a prompt and corresponding reward score
        prompt = prompts[epoch]
        reward_score = rewards[epoch]

        # Tokenize the prompt with padding and truncation
        prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Generate response using the policy model
        # outputs = policy_model.generate(
        #     input_ids=prompt_inputs["input_ids"],
        #     max_length=50,
        #     do_sample=True
        # )

        # # Decode the generated response
        # response = policy_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Tokenize the response with padding and truncation
        response_inputs = tokenizer(real_questions, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Pad queries and responses to the fixed length
        query_tensor = torch.nn.functional.pad(prompt_inputs["input_ids"], (0, fixed_length - prompt_inputs["input_ids"].size(1)), value=tokenizer.pad_token_id).to("cuda")
        response_tensor = torch.nn.functional.pad(response_inputs["input_ids"], (0, fixed_length - response_inputs["input_ids"].size(1)), value=tokenizer.pad_token_id).to("cuda")

        # Collect the inputs and outputs for PPO
        queries.append(query_tensor)
        responses.append(response_tensor)
        scores = [torch.tensor(score).to("cuda" if torch.cuda.is_available() else "cpu") for score in rewards]  # Convert reward list to tensors

        # Perform the PPO step after collecting enough samples
        if (epoch + 1) % ppo_config.batch_size == 0:
            try:
                # Prepare batch lists of queries and responses
                query_batch = [q.squeeze(0) for q in queries]
                response_batch = [r.squeeze(0) for r in responses]
                score_batch = scores[:ppo_config.batch_size]  # Batch of scores

                print(f"Batch shapes before PPO step: Query {len(query_batch)}, Response {len(response_batch)}, Score {len(score_batch)}")

                # Call the PPO step
                stats = ppo_trainer.step(query_batch, response_batch, score_batch)

                # Clear lists after each PPO step
                queries.clear()
                responses.clear()
            except Exception as e:
                print(f"Error during PPO step: {e}")

    # model.save_pretrained(output_name)
    # tokenizer.save_pretrained(output_name)
    # print(f"Model saved to {output_name}")
    return model
