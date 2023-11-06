from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F


class Sbert(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def eval(self):
        self.model.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            out = self.model(input_ids, token_type_ids, attention_mask)
        out = self.mean_pooling(out, attention_mask)
        out = F.normalize(out, p=2, dim=1)
        return out

    def mean_pooling(self, model_output, attention_mask):
        # model_output第0个位置是transformer encoder最后的输出，维度为（B，L，D）
        token_embeddings = model_output[0]
        # input_mask_expanded记录句子哪些位置真的有东西，哪些位置是padding，防止把padding的向量也平均上。
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


# 希望向量化的句子，支持多个句子同时输入
sentences = ["This framework generates embeddings for each input sentence"]
space = "sentence-transformers"
model_name = "distiluse-base-multilingual-cased-v1"

# huggingface接口加载模型
tokenizer = AutoTokenizer.from_pretrained("{}/{}".format(space, model_name))
model = AutoModel.from_pretrained("{}/{}".format(space, model_name))
model.eval()
# 句子token化
encoded_input = tokenizer(
    sentences, padding=True, truncation=True, max_length=512, return_tensors="pt"
)

# sbert = Sbert(model)
# sbert.eval()
# out = sbert(**encoded_input)
# torch.onnx.dynamo_export(sbert, **encoded_input).save("{}.onnx".format(model_name))
torch.onnx.export(
    model,
    (encoded_input["input_ids"], encoded_input["attention_mask"]),
    f="{}.onnx".format(model_name),
    input_names=["input_ids", "attention_mask"],
)

