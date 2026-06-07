# app.py — Garment Attribute Classification
# Gradio demo using two models:
#   - Upper model (v2): sleeve_length, upper_fabric, upper_color
#   - Lower model: lower_fabric, lower_length

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
import gradio as gr

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

UPPER_TASKS = {"sleeve_length": 4, "upper_fabric": 7, "upper_color": 7}
LOWER_TASKS = {"lower_fabric":  6, "lower_length": 4}

UPPER_LABELS = [1, 2]
LOWER_LABELS = [3, 4, 5, 6]

LABEL_MAPS = {
    "sleeve_length": {0:"Sem mangas", 1:"Manga curta", 2:"Manga média", 3:"Manga comprida"},
    "upper_fabric":  {0:"Ganga", 1:"Algodão", 2:"Couro", 3:"Pelo", 4:"Malha", 5:"Chiffon", 6:"Outro"},
    "upper_color":   {0:"Floral", 1:"Estampado", 2:"Riscas", 3:"Cor sólida", 4:"Xadrez", 5:"Outro", 6:"Blocos de cor"},
    "lower_fabric":  {0:"Ganga", 1:"Algodão", 2:"Couro", 3:"Malha", 4:"Chiffon", 5:"Outro"},
    "lower_length":  {0:"Muito curto", 1:"Curto", 2:"Três quartos", 3:"Comprido"},
}

# ── Model architecture ────────────────────────────────────────────────────────
class GarmentClassifier(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        backbone      = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        in_features   = backbone.classifier[1].in_features
        self.heads    = nn.ModuleDict({
            task: nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, n))
            for task, n in tasks.items()
        })

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return {task: head(x) for task, head in self.heads.items()}


class LowerGarmentClassifier(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        backbone      = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        in_features   = backbone.classifier[1].in_features
        self.heads    = nn.ModuleDict({
            task: nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, n))
            for task, n in tasks.items()
        })

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return {task: head(x) for task, head in self.heads.items()}


# ── Load models ───────────────────────────────────────────────────────────────
upper_model = GarmentClassifier(UPPER_TASKS)
ckpt = torch.load("model/best_model_v2.pt", map_location=DEVICE, weights_only=True)
upper_model.load_state_dict(ckpt["model_state_dict"])
upper_model.eval()

lower_model = LowerGarmentClassifier(LOWER_TASKS)
ckpt = torch.load("model/best_lower_model.pt", map_location=DEVICE, weights_only=True)
lower_model.load_state_dict(ckpt["model_state_dict"])
lower_model.eval()

print("Models loaded.")

# ── Transforms ────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── Segmentation crop ─────────────────────────────────────────────────────────
def get_crop(image_np, mask_np, labels):
    """Crop bounding box of region defined by labels. Returns PIL image or None."""
    region = np.isin(mask_np, labels)
    if not region.any():
        return None
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = image_np.copy()
    cropped[~region] = 255
    return Image.fromarray(cropped[rmin:rmax+1, cmin:cmax+1])


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image, segm_mask=None):
    if image is None:
        return None, None, None, None, None

    image_np = np.array(image.convert("RGB"))

    # ── Upper input ───────────────────────────────────────────────────────────
    upper_input = image  # default: full image
    lower_input = image  # default: full image

    if segm_mask is not None:
        mask_np    = np.array(segm_mask.convert("L"))
        upper_crop = get_crop(image_np, mask_np, UPPER_LABELS)
        lower_crop = get_crop(image_np, mask_np, LOWER_LABELS)
        if upper_crop is not None:
            upper_input = upper_crop
        if lower_crop is not None:
            lower_input = lower_crop

    # ── Upper predictions ─────────────────────────────────────────────────────
    if upper_input is None:
        upper_input = image
    if lower_input is None:
        lower_input = image

    img_t = transform(upper_input).unsqueeze(0)
    with torch.no_grad():
        upper_out = upper_model(img_t)

    img_t = transform(lower_input).unsqueeze(0)
    with torch.no_grad():
        lower_out = lower_model(img_t)

    def format_output(outputs, label_maps):
        results = {}
        for task, logits in outputs.items():
            probs  = torch.softmax(logits, dim=1)[0]
            lmap   = label_maps[task]
            results[task] = {lmap[i]: float(probs[i]) for i in range(len(lmap))}
        return results

    upper_results = format_output(upper_out, LABEL_MAPS)
    lower_results = format_output(lower_out, LABEL_MAPS)

    return (
        upper_results["sleeve_length"],
        upper_results["upper_fabric"],
        upper_results["upper_color"],
        lower_results["lower_fabric"],
        lower_results["lower_length"],
    )

# ── Gradio interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Classificação de Atributos de Vestuário") as demo:
    gr.Markdown("<br>", elem_classes=["center"])
    gr.Markdown("# <h1 style='font-size:2.5em'>Classificação de Atributos de Vestuário</h1>", elem_classes=["center"])
    gr.Markdown("Carrega uma imagem de uma pessoa vestida para classificar atributos da peça superior e inferior.", elem_classes=["center"])
    gr.Markdown("*Para melhores resultados, carrega uma imagem de uma só pessoa em formato vertical.*", elem_classes=["center"])
    gr.Markdown("<br>", elem_classes=["center"])

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Imagem")

        with gr.Column(scale=1):
            gr.Markdown("## Peça Superior")
            with gr.Row():
                out_sleeve = gr.Label(label="Manga",  num_top_classes=4)
                out_fabric = gr.Label(label="Tecido", num_top_classes=7)
                out_color  = gr.Label(label="Padrão", num_top_classes=7)

            gr.Markdown("## Peça Inferior")
            with gr.Row():
                out_lower_fabric = gr.Label(label="Tecido",      num_top_classes=6)
                out_lower_length = gr.Label(label="Comprimento", num_top_classes=4)

    def predict_safe(image):
        if image is None:
            return None, None, None, None, None
        return predict(image)

    image_input.change(
        fn=predict_safe,
        inputs=[image_input],
        outputs=[out_sleeve, out_fabric, out_color, out_lower_fabric, out_lower_length]
    )

if __name__ == "__main__":
    demo.launch(css=".center { text-align: center; }")