from transformers import AutoTokenizer, AutoModel
import torch


# 希望向量化的句子，支持多个句子同时输入
sentences = ["This framework generates embeddings for each input sentence"]
space = "sentence-transformers"
model_name = "paraphrase-multilingual-MiniLM-L12-v2"

# huggingface接口加载模型
tokenizer = AutoTokenizer.from_pretrained("{}/{}".format(space, model_name))
model = AutoModel.from_pretrained("{}/{}".format(space, model_name))
model.eval()
# 句子token化
encoded_input = tokenizer(
    sentences, padding=True, truncation=True, max_length=512, return_tensors="pt"
)

torch.onnx.export(
    model,
    (
        encoded_input["input_ids"],
        encoded_input["token_type_ids"],
        encoded_input["attention_mask"],
    ),
    f="{}.onnx".format(model_name),
    input_names=["input_ids", "token_type_ids", "attention_mask"],
)
