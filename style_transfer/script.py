import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

device = torch.device("cuda")

vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad_(False)

# Non-overlapping VGG slices
slice1 = vgg[:4]    # relu1_2
slice2 = vgg[4:9]   # relu2_2
slice3 = vgg[9:18]  # relu3_4

def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

def gram(x):
    B, C, H, W = x.shape
    f = x.view(B, C, H*W)
    return torch.bmm(f, f.transpose(1,2)) / (H * W)

def get_style_grams(x):
    h1 = slice1(x)
    h2 = slice2(h1)
    h3 = slice3(h2)
    return gram(h1), gram(h2), gram(h3)

# Resize to manageable size before everything
MAX_SIZE = 512
to_tensor = T.ToTensor()

def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return to_tensor(img).unsqueeze(0).to(device)

content = load_image("/mnt/ssd2/ualibek/example/DSC07956.JPG")
style   = load_image("/mnt/ssd2/ualibek/style_images/van_gogh.jpg")

with torch.no_grad():
    style_grams = get_style_grams(normalize(style))
    style_grams = [g.detach() for g in style_grams]

canvas = content.clone().detach().requires_grad_(True)
optimizer = torch.optim.LBFGS([canvas], lr=1.0, max_iter=20)

step = 0
for _ in range(50):  # 100 LBFGS steps
    def closure():
        optimizer.zero_grad()
        gen_grams = get_style_grams(normalize(canvas))
        loss = sum((g - s).pow(2).mean() for g, s in zip(gen_grams, style_grams))
        loss.backward()
        with torch.no_grad():
            canvas.data.clamp_(0, 1)
        return loss

    loss = optimizer.step(closure)
    step += 1
    if step % 10 == 0:
        print(f"step {step} loss={loss.item():.4e} "
              f"pixel_change={( canvas - content).abs().mean().item():.4f}")

vutils.save_image(canvas.detach(), "output.jpg")
print("Saved output.jpg")