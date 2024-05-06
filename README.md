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

## 用模型微調訓練方法
1.上傳如下的json檔

```python
  {
    "instruction": "如何减少空气污染？",
    "input": "",
    "output": "有很多方法来减少空气污染:\n\n1. 减少机动车排放: 通过支持可再生能源来驱动的交通工具，如电动汽车、混合动力汽车，使用公共交通工具，骑自行车或步行。\n2. 鼓励清洁能源: 政府和企业可以通过提供优惠政策和补贴来鼓励人们使用清洁能源。\n3. 减少工业排放: 工厂可以通过采用更清洁的生产方法，使用减排技术来减少工业废气的排放。\n4. 绿化城市: 种植更多的树木和植物可以吸收空气中的污染物和二氧化碳，有助于减少空气污染。\n5. 减少家庭能耗: 通过使用节能电器、关闭不必要的电源开关、减少用电量，能够减少家庭用电量，从而减少空气污染。\n6. 垃圾分类和回收利用: 妥善处理垃圾，避免焚烧垃圾，对垃圾进行分类和回收利用，可以减少污染物排放。\n\n总之，减少空气污染需要政府、企业和个人共同努力。我们每个人都要承担责任，通过自己的实际行动来保护环境。"
  },
  {
    "instruction": "辨认不同之处。 ",
    "input": "输入：Twitter，Instagram，Telegram。",
    "output": "Twitter，Instagram 和 Telegram 都是流行的社交媒体平台，但它们之间也有一些不同之处：\n\n- Twitter: 是一个微博社交媒体平台，用户可以发布 280 字符以内的消息（即推文）并与其他用户互动。它主要关注文字交流，尽管用户也可以分享照片和视频。\n- Instagram: 是一个图片和视频分享的社交媒体平台，用户可以在 Instagram 上发布照片和视频并与其他用户互动。它主要关注图片和视频的分享，尽管用户也可以添加文字说明。\n- Telegram: 是一个即时通讯和社交媒体平台，用户可以通过文本、语音、图片和视频与其他用户即时交流。与 Twitter 和 Instagram 不同，Telegram 更侧重于即时通讯，它还提供了端到端加密的私人聊天功能。"
  },
```
2. python main.py 來訓練

3. 測試結果
   ![image](https://github.com/j40pl7llyccl/Fine-tuning_LLM_liu/assets/24970006/0848698b-08e3-4bc5-ab8a-74cd8044d040)


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
