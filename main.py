from data_processing import preprocess_data
from model_training import train_model

if __name__ == "__main__":
    data_file = './huanhuan.json'
    model_name = "THUDM/chatglm3-6b"
    output_dir = "./output/ChatGLM"
    tokenized_ds = preprocess_data(data_file, tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True))
    train_model(tokenized_ds, model_name, output_dir)