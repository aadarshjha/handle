import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type="divided_space_time",
    pretrained_model="D:\Projects\ML\models\stformer.pyth",
).cuda()

dummy_video = torch.randn(
    2, 3, 8, 224, 224
).cuda()  # (batch x channels x frames x height x width)

pred = model(
    dummy_video,
)  # (2, 400)

for param in model.parameters():
    param.requires_grad = False

model.model.head.weight.requires_grad = True
model.model.head.bias.requires_grad = True

print(model)
