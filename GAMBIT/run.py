import os
import torch
import time
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
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
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, TrainingArguments, Trainer, AutoModel
import argparse

from utils.finetune_Detective import *
from utils.finetune_Cheater import *
from utils.model_setup import *
from utils.generate_answer import *
from utils.generate_question import *
from utils.response_filter import *
from utils.detection import *
from utils.finetune_pipeline import *


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"GAMBIT\skeleton_model"


def download_skeleton_model(model_name):
    skeleton_folder = MODEL_PATH
    os.makedirs(skeleton_folder, exist_ok=True)  # Ensure the folder exists
    
    model_path = os.path.join(skeleton_folder, model_name)
    
    if not os.path.exists(model_path):  
        print(f"Downloading skeleton model: {model_name}...")
        
        # Load and save the model from Hugging Face
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(model_path)
        
        print(f"Model saved in: {model_path}")
    else:
        print(f"Skeleton model already exists at: {model_path}")

    return model_path

def conversation_workflow(model_name:str, question_model_name:str, detection_model_name:str, model_path:str, num_rounds=15, max_cluster_size=15):
    print("1")
    max_gen_cluster_size=50
    model1, tokenizer1 = setup_question_model(question_model_name)
    model2, tokenizer2 = setup_quantized_model(model_name)
    
    detection_model, detection_tokenizer = setup_detection_model(detection_model_name)

    print("2")
    Cheater_conversations = []
    Cheater_cluster = []
    Gen_conversations = []
    Gen_cluster = []
    confidence_history = []
    real_questions=[]
    real_questions_gen_passed=[]
    confidence_reward = []
    
    initial_data = collect_initial_data(
                    model1, tokenizer1,
                    model2, tokenizer2,
                    detection_model, detection_tokenizer
                )
    print("3")
    # Create a directory to store results if it doesn't exist
    os.makedirs('results', exist_ok=True)

    for round in range(num_rounds):
        print(f"Conversation Round Set {round + 1}")

        while True:
            # Generate question

            real_questions_val = generate_question(model1, tokenizer1, "Ask an interesting question:")
            question = "You are a human being and you need to answer this question in the most human like way possible \nQ: " + real_questions_val + "\nA:"
            print(f"Model1's Q: {question}")
            real_questions.append(real_questions_val)
            
            # Generate five response variants and select the best one
            responses = generate_response(model2, tokenizer2, question,num_variants=4)
            best_answer, confidence = select_best_response(detection_model, detection_tokenizer, responses)
            print(f"Selected Answer: {best_answer} (Confidence: {confidence})")
            confidence_history.append(confidence)


            worst_answer, confidence2 = select_worst_response(detection_model, detection_tokenizer, responses)
            print(f"Worst Answer: {worst_answer} (Confidence: {confidence2})")

            # Always add the confidence score, regardless of threshold
            confidence_history.append(confidence2)

            # Calculate running average of confidence scores
            running_avg_confidence = sum(confidence_history) / len(confidence_history)
            print(f"Running Average Confidence: {running_avg_confidence}")

            # Store the entry
            if(confidence < running_avg_confidence-0.07):
                Cheater_entry = {
                    'round': round + 1,
                    'question': question,
                    'answer': best_answer,
                    'detection_confidence': confidence,
                    'running_avg_confidence': running_avg_confidence
                }
                Cheater_cluster.append(Cheater_entry)
                Cheater_conversations.append(Cheater_entry)

            if(confidence > 0.99):
                Gen_entry = {
                    'round': round + 1,
                    'question': question,
                    'answer': worst_answer,
                    'detection_confidence': confidence2,
                    'running_avg_confidence': running_avg_confidence
                }
                real_questions_gen_passed.append(real_questions[-1])
                confidence_reward.append(confidence)
                Gen_cluster.append(Gen_entry)
                Gen_conversations.append(Gen_entry)


            print(f"Added to cluster. Current cluster size: {len(Cheater_cluster)}")

            # When cluster is full, fine-tune the model
            if len(Gen_cluster) == max_gen_cluster_size:
                word2vec_model = setup_embedding_model(initial_data)
                model1 = finetune_detective_with_RLHF(model1, tokenizer1,real_questions_gen_passed,word2vec_model=word2vec_model)
                
                real_questions_gen_passed = []
                Gen_cluster=[]

            if len(Cheater_cluster) == max_cluster_size:

                # Create a DataFrame for fine-tuning
                conversation_df = pd.DataFrame(Cheater_cluster)
                conversation_df['text'] = conversation_df.apply(lambda row: f"Q: {row['question']} A: {row['answer']}", axis=1)

                # Save the dataset
                dataset_filename = f'./results/formatted_dataset_round_{round+1}.csv'
                conversation_df[['text']].to_csv(dataset_filename, index=False)

                # Create a comprehensive results DataFrame
                results_df = pd.DataFrame(Cheater_cluster)
                results_df['running_avg_confidence'] = running_avg_confidence
                results_filename = f'./results/conversation_results_round_{round+1}.csv'
                results_df.to_csv(results_filename, index=False)

                if(round == num_rounds-2):
                    model2 = fine_tune_model(model2, tokenizer2, dataset_filename,MODEL_PATH=model_path, output_name=f'model2_round_{round+1}')
                    break
                model2 = just_fine_tune_model(model2, tokenizer2, dataset_filename, f'model2_round_{round+1}')

                # Reset the cluster
                Cheater_cluster = []
                break

    # Save final conversations with running average
    final_df = pd.DataFrame(Cheater_conversations)
    final_df.to_csv('./results/final_conversations_with_detection.csv', index=False)
    return final_df

def main():
    parser = argparse.ArgumentParser(description="Run model processing in GAMBIT")
    parser.add_argument("--model", nargs="+", required=True, help="Model names")
    
    args = parser.parse_args()
    selected_models = args.model
    model_name = selected_models[0]
    question_model_name = selected_models[1]
    detection_model_name = selected_models[2]
    model_path = download_skeleton_model(model_name)
    results_df = conversation_workflow(model_name,question_model_name,detection_model_name,model_path)
    print(results_df)
    
if __name__ == "__main__":
    main()