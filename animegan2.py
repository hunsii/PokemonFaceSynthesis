from PIL import Image
import torch
# import gradio as gr



model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cpu",
    progress=False
)


# model1 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1",  device="cuda")
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, device="cpu",side_by_side=False
)
import glob

l = glob.glob("images/*/*.png")
img = Image.open(l[232]).convert("RGB").resize((512,512))
img  = Image.open("test.png").convert("RGB").resize((512,512))
img.save("input.png")
out = face2paint(model2, img)
out.save("result.png")
  