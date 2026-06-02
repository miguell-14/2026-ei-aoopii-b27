# app.py — Garment Attribute Classification
# Gradio demo using EfficientNet-B0 with 3 classification heads

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "model/best_model_v2.pt"
DEVICE     = torch.device("cpu")  # use CPU for local inference

TASKS = {
    "sleeve_length": 4,
    "upper_fabric":  7,
    "upper_color":   7,
}

LABEL_MAPS = {
    "sleeve_length": {0:"Sem mangas", 1:"Manga curta", 2:"Manga meia", 3:"Manga comprida"},
    "upper_fabric":  {0:"Ganga", 1:"Algodão", 2:"Couro", 3:"Pelo", 4:"Malha", 5:"Chiffon", 6:"Outro"},
    "upper_color":   {0:"Floral", 1:"Estampado", 2:"Riscas", 3:"Cor sólida", 4:"Xadrez", 5:"Outro", 6:"Blocos de cor"},
}

# ── Model ─────────────────────────────────────────────────────────────────────
class GarmentClassifier(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        backbone      = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        in_features   = backbone.classifier[1].in_features

        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
            for task, num_classes in tasks.items()
        })

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return {task: head(x) for task, head in self.heads.items()}


# ── Load model ────────────────────────────────────────────────────────────────
model = GarmentClassifier(TASKS)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── Inference function ────────────────────────────────────────────────────────
def predict(image):
    img    = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)

    results = {}
    for task, logits in outputs.items():
        probs      = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        pred_label = LABEL_MAPS[task][pred_idx]
        confidence = probs[pred_idx].item()

        results[task] = {
            label: float(probs[i]) for i, label in LABEL_MAPS[task].items()
        }

    sleeve = LABEL_MAPS["sleeve_length"][outputs["sleeve_length"].argmax(dim=1).item()]
    fabric = LABEL_MAPS["upper_fabric"][outputs["upper_fabric"].argmax(dim=1).item()]
    color  = LABEL_MAPS["upper_color"][outputs["upper_color"].argmax(dim=1).item()]

    return (
        results["sleeve_length"],
        results["upper_fabric"],
        results["upper_color"],
    )

# ── Gradio interface ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Carrega uma imagem de roupa"),
    outputs=[
        gr.Label(label="Comprimento da manga"),
        gr.Label(label="Tipo de tecido"),
        gr.Label(label="Padrão de cor"),
    ],
    title="Classificação de Atributos de Vestuário",
    description="Carrega uma imagem de uma pessoa vestida para classificar o comprimento da manga, tipo de tecido e padrão de cor.",
    flagging_mode="never",
)
if __name__ == "__main__":
    demo.launch()