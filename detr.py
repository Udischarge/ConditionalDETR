import colorsys
import os
import time
import seaborn

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from matplotlib.colors import ListedColormap
from torchvision.transforms import ToPILImage

from models.conditional_detr import ConditionalDETR
from util.utils import (
    cvtColor,
    get_classes,
    preprocess_input,
    resize_image,
    detr_resize_image,
    show_config,
)
from util.utils_bbox import DecodeBox, DecodeBox_heat

"""
训练自己的数据集必看注释！
"""

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, images_shape):
    img_w, img_h = images_shape[0]
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(b.device)
    return b

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
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


class Detection_Transformers(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": "/workspace/ConditionalDETR/ConditionalDETR_r50_epoch50.pth",
        "classes_path": "/workspace/Detect/GanDetect/model_data/coco_classes.txt",
        # ---------------------------------------------------------------------#
        #   输入图片的大小
        # ---------------------------------------------------------------------#
        "min_length": 800,
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   主干网络的种类
        # ---------------------------------------------------------------------#
        "backbone": "resnet50",
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化detr
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox_heat()

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(
                lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors,
            )
        )
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        # ---------------------------------------------------#
        #   建立detr模型，载入detr模型的权重
        # ---------------------------------------------------#
        self.net = ConditionalDETR(self.backbone, "sine", 256, self.num_classes, num_queries=100)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print("{} model, anchors, and classes loaded.".format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = detr_resize_image(image, self.min_length)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            _, outputs = self.net(images)
            results, _ = self.bbox_util(outputs, images_shape, self.confidence)

            if results[0] is None:
                return image

            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype="int32")
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(
            font="model_data/simhei.ttf",
            size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"),
        )
        thickness = int(max((image.size[0] + image.size[1]) // self.min_length, 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype("int32"))
                left = max(0, np.floor(left).astype("int32"))
                bottom = min(image.size[1], np.floor(bottom).astype("int32"))
                right = min(image.size[0], np.floor(right).astype("int32"))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(
                    os.path.join(dir_save_path, "crop_" + str(i) + ".png"),
                    quality=95,
                    subsampling=0,
                )
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom).astype("int32"))
            right = min(image.size[0], np.floor(right).astype("int32"))

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode("utf-8")
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i], outline=self.colors[c]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c],
            )
            draw.text(text_origin, str(label, "UTF-8"), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, self.min_length)
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()

            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
                if self.cuda:
                    images = images.cuda()
                    images_shape = images_shape.cuda()

                outputs = self.net(images)
                results, _ = self.bbox_util(outputs, images_shape, self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx

        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to(
            "cpu"
        )  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f"Starting export with onnx {onnx.__version__}.")
        torch.onnx.export(
            self.net,
            im,
            f=model_path,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=input_layer_names,
            output_names=output_layer_names,
            dynamic_axes=None,
        )

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim

            print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
            model_onnx, check = onnxsim.simplify(
                model_onnx, dynamic_input_shape=False, input_shapes=None
            )
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)

        print("Onnx model save as {}".format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(
            os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w"
        )
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = detr_resize_image(image, self.min_length)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            _, outputs = self.net(images)
            results, _ = self.bbox_util(outputs, images_shape, self.confidence)

            if results[0] is None:
                return

            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype="int32")
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write(
                "%s %s %s %s %s %s\n"
                % (
                    predicted_class,
                    score[:6],
                    str(int(left)),
                    str(int(top)),
                    str(int(right)),
                    str(int(bottom)),
                )
            )

        f.close()
        return

    def heat_map(self, image):
        # 获取训练好的参数
        for name, parameters in self.net.module.named_parameters():
            # 获取训练好的object queries，即pq:[100,256]
            if name == "query_embed.weight":
                pq = parameters
            # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[256*3,256]，[256*3] qkv三部分，所以是*3
            if name == "transformer.decoder.layers.5.multihead_attn.in_proj_weight":
                in_proj_weight = parameters
            if name == "transformer.decoder.layers.5.multihead_attn.in_proj_bias":
                in_proj_bias = parameters

        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#

        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = detr_resize_image(image, self.min_length)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)
            ),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_shape = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images = images.cuda()
                images_shape = images_shape.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            _, outputs = self.net(images)
            results, query_id = self.bbox_util(outputs, images_shape, confidence=0.6)

            if results[0] is None:
                return image
            query_id = query_id[0]
            _results = results[0].cpu().numpy()
            top_label = np.array(_results[:, 5], dtype="int32")
            top_conf = _results[:, 4]
            top_boxes = _results[:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(
            font="model_data/simhei.ttf",
            size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"),
        )
        thickness = int(max((image.size[0] + image.size[1]) // self.min_length, 1))

        #  准备存储前馈该图片时的值
        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        cq = []  # 存储detr中的 cq
        pk = []  # 存储detr中的 encoder pos
        memory = []  # 编码器最后一层的输入/解码器的输入特征

        # 注册hook
        hooks = [
            # 获取resnet最后一层特征图 [1, 2048, H/32, W/32]
            self.net.module.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            # 获取encoder的图像特征图memory [H/32 * W/32, 1, 256]
            self.net.module.transformer.encoder.register_forward_hook(
                lambda self, input, output: memory.append(output)
            ),
            # 获取encoder的最后一层layer的self-attn weights [1, H/32 * W/32, H/32 * W/32]
            self.net.module.transformer.encoder.layers[
                -1
            ].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            # 获取decoder的最后一层layer中交叉注意力的 weights [1, 100, H/32 * W/32]
            self.net.module.transformer.decoder.layers[
                -1
            ].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
            # 获取decoder最后一层self-attn的输出cq [100, 1, 256]
            self.net.module.transformer.decoder.layers[-1].norm1.register_forward_hook(
                lambda self, input, output: cq.append(output)
            ),
            # 获取图像特征图的位置编码pk [1, 256, H/32, W/32]
            self.net.module.backbone[-1].register_forward_hook(
                lambda self, input, output: pk.append(output)
            ),
        ]

        # propagate through the model
        _ = self.net(images)

        # 用完的hook后删除
        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]  # [1,2048,25,34]
        enc_attn_weights = enc_attn_weights[0]  # [1,850,850]   : [N,L,S]
        # [1,100,850]: [N,L,S]-->[batch, tgt_len, src_len]
        dec_attn_weights = dec_attn_weights[0]
        memory = memory[0]  # [850,1,256] # 编码器最后一层的输入/解码器的输入特征

        cq = cq[0]  # decoder的self_attn:最后一层输出[100,1,256]
        pk = pk[0]  # [1,256,25,34]

        # 求attn_output_weights以绘制各个head的注意力权重
        pk = pk.flatten(-2).permute(2, 0, 1)  # [1,256,850] --> [850,1,256]
        pq = pq.unsqueeze(1).repeat(1, 1, 1)  # [100,1,256]、

        #todo 这里可以选择Q和K  Q: cq + pq  K: memory + pk 可以自由组合
        # q = torch.concat([cq, pq], dim=2)
        q = pq + cq
        # q = cq
        # k = torch.concat([memory, pk], dim=2)
        k = memory + pk
        # k = memory

        # 将q和k完成线性层的映射，代码参考自nn.MultiHeadAttn()
        _b = in_proj_bias
        _start = 0
        _end = 256
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = nn.functional.linear(q, _w, _b)

        _b = in_proj_bias
        _start = 256
        _end = 256 * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = nn.functional.linear(k, _w, _b)

        scaling = float(256) ** -0.5
        q = q * scaling
        q = q.contiguous().view(100, 8, 32).transpose(0, 1)
        k = k.contiguous().view(-1, 8, 32).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        attn_output_weights = attn_output_weights.view(1, 8, 100, -1)
        attn_output_weights = attn_output_weights.view(1 * 8, 100, -1)
        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = attn_output_weights.view(1, 8, 100, -1)

        h, w = conv_features["0"].tensors.shape[-2:]

        # 后续可视化各个头
        attn_every_heads = attn_output_weights  # [1,8,100,850]
        attn_output_weights = attn_output_weights.sum(dim=1)/8  # [1,100,850]

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        figs = []
        for i, c in list(enumerate(top_label)):
            # 创建一个新的图像
            # fig, axs = plt.subplots(1,2,figsize=(15, 7))
            fig, axs = plt.subplots(3,3,figsize=(15, 15))
            image_copy = image.copy()

            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            attn_weights_np = (
                dec_attn_weights[0][query_id[i]].view(h, w).detach().cpu().numpy()
            )
            for head_id in range(0,8):
                attn_every_heads_np = attn_every_heads[0][head_id][query_id[i]].view(h, w).detach().cpu().numpy()
                #  当k = memory + pk，且8个头取平均值时，就得到了交叉注意力热力图
                # attn_output_weights_np = attn_output_weights[0][query_id[i]].view(h, w).detach().cpu().numpy()
                axs[head_id // 3, head_id % 3].imshow(attn_every_heads_np)
                axs[head_id // 3, head_id % 3].axis('off')
                axs[head_id // 3, head_id % 3].set_title(f'head:{head_id}',fontsize = 35)

            #  当k = memory + pk，且8个头取平均值时，就得到了交叉注意力热力图
            # attn_output_weights_np = attn_output_weights[0][query_id[i]].view(h, w).detach().cpu().numpy()
            # axs[0].imshow(attn_output_weights_np)
            # axs[0].axis('off')
            # axs[0].set_title('head:sum',fontsize = 35)
            # 计算缩放比例
            zoom_ratio = [image.size[1] / h, image.size[0] / w]
            # 调整热力图的尺寸
            attn_weights_np = zoom(attn_weights_np, zoom_ratio)

            # # 设置热力图的值小于一定值的部分为NaN
            # threshold = 0.01  # 你可以根据需要调整这个阈值
            # attn_weights_np[attn_weights_np < threshold] = np.nan

            rgb = tuple([x / 255 for x in self.colors[c]])
            cmap = mcolors.LinearSegmentedColormap.from_list("", [(0, 0, 0, 0), rgb])
            seaborn.heatmap(attn_weights_np, alpha=0.5, ax=axs[2,2], cmap=cmap, cbar=False)
            axs[2,2].axis("off")  # 隐藏坐标轴

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype("int32"))
            left = max(0, np.floor(left).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom).astype("int32"))
            right = min(image.size[0], np.floor(right).astype("int32"))

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image_copy)
            label_size = draw.textsize(label, font)
            label = label.encode("utf-8")
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j], outline=self.colors[c]
                )
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c],
            )
            draw.text(text_origin, str(label, "UTF-8"), fill=(0, 0, 0), font=font)

            axs[2,2].imshow(image_copy, alpha=1)
            axs[2,2].axis("off")  # 隐藏坐标轴
            fig.tight_layout()        # 自动调整子图来使其填充整个画布
            # plt.savefig(f"{heatmap_save_path}/heatmap{i}.png")
            figs.append(fig)
        return figs

