import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from config.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-uk"):
        print("Initializing transformer model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def train(self, dataset, epochs=1, learning_rate=2e-5):
        """Proper training method implementation"""
        print("Tokenizing dataset...")
        
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['EN'],
                padding='max_length',
                truncation=True,
                max_length=512
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['UK'],
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
            inputs['labels'] = labels['input_ids']
            return inputs
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['EN', 'UK']
        )

        print("Setting up training...")
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,  
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_dir="./logs",
            logging_steps=100,
            fp16=True,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
        )

        print("Starting training...")
        trainer.train()
        print("Training complete!")