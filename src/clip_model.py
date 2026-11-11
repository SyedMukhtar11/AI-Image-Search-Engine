
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPModelManager:

    def __init__(self, model_name="openai/clip-vit-base-patch32"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model = CLIPModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    def get_text_embedding(self, text):

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

            if hasattr(features, "pooler_output"):
                features = features.pooler_output

            features = F.normalize(features, p=2, dim=-1)

        return features.cpu().numpy()
