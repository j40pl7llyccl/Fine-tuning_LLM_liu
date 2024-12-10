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
    "instruction": "如何減少空氣污染？",
    "input": "",
    "output": "有很多方法來減少空氣污染:\n\n1. 減少機動車排放: 透過支持再生能源來驅動的交通工具，如電動車、混合動力汽車，使用公共交通工具，騎自行車或步行。 \n2. 鼓勵清潔能源: 政府和企業可以透過提供優惠政策和補貼來鼓勵人們使用清潔能源。 \n3. 減少工業排放: 工廠可以透過採用更清潔的生產方法，使用減排技術來減少工業廢氣的排放。 \n4. 綠化城市: 種植更多的樹木和植物可以吸收空氣中的污染物和二氧化碳，有助於減少空氣污染。 \n5. 減少家庭能耗: 透過使用節能電器、關閉不必要的電源開關、減少用電量，能夠減少家庭用電量，從而減少空氣污染。 \n6. 垃圾分類和回收： 妥善處理垃圾，避免焚燒垃圾，對垃圾進行分類和回收利用，可以減少污染物排放。 \n\n總之，減少空氣污染需要政府、企業和個人共同努力。我們每個人都要承擔責任，透過自己的實際行動來保護環境。"
  },
  {
    "instruction": "辨認不同之處。 ",
    "input": "输入：Twitter，Instagram，Telegram。",
    "output": "Twitter，Instagram 和Telegram 都是流行的社交媒體平台，但它們之間也有一些不同之處：\n\n- Twitter: 是一個微博社交媒體平台，用戶可以發布280 字符以內的消息（即推文）並與其他使用者互動。它主要關注文字交流，儘管用戶也可以分享照片和影片。 \n- Instagram: 是一個圖片和影片分享的社群媒體平台，用戶可以在 Instagram 上發布照片和影片並與其他用戶互動。它主要關注圖片和影片的分享，儘管用戶也可以添加文字說明。 \n- Telegram: 是一個即時通訊和社交媒體平台，用戶可以透過文字、語音、圖片和影片與其他用戶即時交流。與 Twitter 和 Instagram 不同，Telegram 更專注於即時通訊，它還提供了端對端加密的私人聊天功能。"
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
