#model_training.py
import torch
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import TaskType, get_peft_model, LoraConfig
import matplotlib.pyplot as plt

def train_model(tokenized_ds, model_name, output_dir, epochs=1, batch_size=1, gradient_accumulation_steps=8, logging_steps=20):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.half, trust_remote_code=True, low_cpu_mem_usage=True)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"query_key_value"}, r=8, lora_alpha=32)
    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        num_train_epochs=epochs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=None, padding=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
        callbacks=[TrainingCallback()] ,
    )

    train_loss = trainer.train()
    plot_loss(train_loss.log_history)  
    
class TrainingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.train_losses.append(logs.get("loss"))

def plot_loss(log_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(log_history)), log_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
