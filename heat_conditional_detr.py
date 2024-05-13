# #------------------------------------------------------------#
# 可视化Detr方法：
# spatial attention weight : (cq + oq)*pk
# combined attention weight: (cq + oq)*(memory + pk)
# 其中:
#     pk:原始特征图的位置编码;
#     oq:训练好的object queries
#     cq:decoder最后一层self-attn中的输出query
#     memory:encoder的输出
# #------------------------------------------------------------#
# 在此基础上只要稍微修改便可可视化ConditionalDetr的Fig1特征图
# #------------------------------------------------------------#
# 代码参考自:https://github.com/facebookresearch/detr/tree/colab
# #------------------------------------------------------------#

import math
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.nn.functional import dropout,linear,softmax
torch.set_grad_enabled(False)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 加载线上的模型
model = torch.hub.load('../ConditionalDETR', 'conditional_detr_resnet50', source='local',pretrained=True)
model.eval()
# 获取训练好的参数
# for name, parameters in model.named_parameters():
    # 获取训练好的object queries，即pq:[100,256]
    # print(name, parameters.shape)
    # if name == 'query_embed.weight':
    #     pq = parameters
    # # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[768]
    # if name == 'transformer.decoder.layers.5.ca_qcontent_proj.weight':
    #     ca_qcontent_proj_weight = parameters
    # if name == 'transformer.decoder.layers.5.ca_qcontent_proj.bias':
    #     ca_qcontent_proj_bias = parameters
# 线上下载图像
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# im = Image.open(requests.get(url, stream=True).raw)
img_path = '/workspace/ConditionalDETR/000000039769.jpg'
im = Image.open(img_path)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
# keep = probas.max(-1).values > 0
keep = probas.max(-1).values > 0.5

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []
cq = []     # 存储detr中的 cq
pq =[]
pk =  []    # 存储detr中的 encoder pos
ck =[]
# 注册hook
hooks = [
    # 获取resnet最后一层特征图
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    # 获取encoder的最后一层layer的self-attn weights
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    # 获取decoder的最后一层layer中交叉注意力的 weights
    model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
# ---------------------------------------------------------------------------------#
    # 获取encoder的图像特征图memory的映射ck
    model.transformer.decoder.layers[-1].ca_kcontent_proj.register_forward_hook(
        lambda self, input, output: ck.append(output)
    ),
    # 获取decoder最后一层self-attn的输出的映射cq
    model.transformer.decoder.layers[-1].ca_qcontent_proj.register_forward_hook(
        lambda self, input, output: cq.append(output)
    ),
    # 获取图像特征图的位置编码pk
    model.transformer.decoder.layers[-1].ca_kpos_proj.register_forward_hook(
        lambda self, input, output: pk.append(output)
    ),
    # 获取图像特征图的位置编码pq
    model.transformer.decoder.layers[-1].ca_qpos_sine_proj.register_forward_hook(
        lambda self, input, output: pq.append(output)
    ),
]

# propagate through the model
outputs = model(img)

# 用完的hook后删除
for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]       # [1,2048,25,34]
enc_attn_weights = enc_attn_weights[0] # [1,850,850]   : [N,L,S]
dec_attn_weights = dec_attn_weights[0] # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]

cq = cq[0]    # decoder的self_attn:最后一层输出[100,1,256]
ck = ck[0]
pq = pq[0]
pk = pk[0]    # [1,256,25,34]

num_queries = 300
bs = 1
n_model = 256
nhead = 8
h, w = conv_features['0'].tensors.shape[-2:]
hw = h * w

cq = cq.view(num_queries, bs, nhead, n_model//nhead)
pq = pq.view(num_queries, bs, nhead, n_model//nhead)
# q = cq.view(num_queries, bs, n_model)
q = torch.cat([cq, pq], dim=3).view(num_queries, bs, n_model * 2)
ck = ck.view(hw, bs, nhead, n_model//nhead)
pk = pk.view(hw, bs, nhead, n_model//nhead)
# k = ck.view(hw, bs, n_model)
k = torch.cat([ck, pk], dim=3).view(hw, bs, n_model * 2)
#------------------------------------------------------#


scaling = float(256) ** -0.5
q = q * scaling * 2
q = q.contiguous().view(300, 8, 64).transpose(0, 1)
k = k.contiguous().view(-1, 8, 64).transpose(0, 1)
attn_output_weights = torch.bmm(q, k.transpose(1, 2))


attn_output_weights = attn_output_weights.view(1, 8, 300, hw)
attn_output_weights = attn_output_weights.view(1 * 8,300, hw)
attn_output_weights = softmax(attn_output_weights, dim=-1) # [1,100,850]
attn_output_weights = attn_output_weights.view(1, 8, 300, hw)

# 后续可视化各个头
attn_every_heads = attn_output_weights # [1,8,100,850]
attn_sum_heads = attn_output_weights.sum(dim=1) / 8 # [1,100,850]

#-----------#
#   可视化
#-----------#
# get the feature map shape


fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=11, figsize=(11, 20))  # [11,14]
colors = COLORS * 100

# 可视化
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    # 可视化decoder的注意力权重
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}',fontsize = 20)
    # 可视化框和类别
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()],fontsize = 20)
    # 分别可视化8个头部的位置特征图
    for head in range(2, 2 + 8):
        ax = ax_i[head]
        ax.imshow(attn_every_heads[0, head-2, idx].view(h,w))
        ax.axis('off')
        ax.set_title(f'head:{head-2}',fontsize = 20)
    ax = ax_i[10]
    ax.imshow(attn_sum_heads[0, idx].view(h,w))
    ax.axis('off')
    ax.set_title(f'sum_head',fontsize = 20)

fig.tight_layout()        # 自动调整子图来使其填充整个画布
plt.show()
# 保存图片
plt.savefig('detr_attention_weights0.jpg')