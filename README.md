# Fine-tuning_LLM_liu
結合兩個數據庫來做微調模型來達到知識問答和聊天的機器人
- wikimedia/wikipedia
- stingning/ultrachat

項目URL: [Large language model fine-tuning(基於mistral-7B微調的LLM模型)](https://huggingface.co/j40pl7lly/fine-tuning-chat-liu)

1.效率：透過使用GPU加速、LoRA、梯度累積和混合精度訓練（FP16），最大化運算資源和訓練速度。

2.適應性：透過LoRA對模型的特定組件進行微調，它可以以減少參數達到(30%)以更新更有效地適應目標任務的預訓練模型。

## api使用方法:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("j40pl7lly/fine-tuning-chat-liu")
model = AutoModelForCausalLM.from_pretrained("j40pl7lly/fine-tuning-chat-liu")
```

## Reference
If you use this model and love it, use this to cite it 🤗

## Citation

```
@misc{privacy_faceemotionrecognition_system,
      title={Fine-tuned LLM model based on open source mistral-7B},
      author={Liu Hsin Kuo},
      year={2024},
}
```
