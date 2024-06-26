import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, boxes


class DecodeBox_heat(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, confidence):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        img_h = img_h.float()
        img_w = img_w.float()
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        outputs = torch.cat(
            [
                torch.unsqueeze(boxes[:, :, 1], -1),
                torch.unsqueeze(boxes[:, :, 0], -1),
                torch.unsqueeze(boxes[:, :, 3], -1),
                torch.unsqueeze(boxes[:, :, 2], -1),
                torch.unsqueeze(scores, -1),
                torch.unsqueeze(labels.float(), -1),
            ],
            -1,
        )

        results = []
        indices = []
        for output in outputs:
            mask = output[:, 4] > confidence
            results.append(output[mask])
            indices.append(
                torch.nonzero(mask).squeeze(-1)
            )  # 获取满足条件的结果在output中的索引
        return results, indices


class DecodeBox(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, confidence):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        img_h = img_h.float()
        img_w = img_w.float()
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        outputs = torch.cat(
            [
                torch.unsqueeze(boxes[:, :, 1], -1),
                torch.unsqueeze(boxes[:, :, 0], -1),
                torch.unsqueeze(boxes[:, :, 3], -1),
                torch.unsqueeze(boxes[:, :, 2], -1),
                torch.unsqueeze(scores, -1),
                torch.unsqueeze(labels.float(), -1),
            ],
            -1,
        )

        results = []
        query_id = []
        for output in outputs:
            results.append(output[output[:, 4] > confidence])
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


# 之前为了不失真的resize，增加了灰调，因此要更正预测结果
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.0)
    box_maxes = box_yx + (box_hw / 2.0)
    boxes = np.concatenate(
        [
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2],
        ],
        axis=-1,
    )
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


#! 这里output不是三个了，就一个[batch_size, num_class+1+4, 40, 40]
def decode_outputs(outputs, input_shape):
    # 如果output不是列表，就转换成列表
    if not isinstance(outputs, list):
        outputs = [outputs]
    grids = []
    strides = []
    hw = [x.shape[-2:] for x in outputs]
    # ---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    # ---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(
        0, 2, 1
    )
    # ---------------------------------------------------#
    #   获得每一个特征点属于每一个种类以及是否存在物体的概率
    # ---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        # ---------------------------#
        #   根据特征层的高宽生成网格点
        # ---------------------------#
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # ---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        # ---------------------------#
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    # ---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    # ---------------------------#
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    # ------------------------#
    #   根据网格点进行解码
    # ------------------------#
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # -----------------#
    #   归一化
    # -----------------#
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression(
    prediction,
    num_classes,
    input_shape,
    image_shape,
    letterbox_image,
    conf_thres=0.5,
    nms_thres=0.4,
):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        output[i] = detections[nms_out_index]

        # 非极大值抑制 看一下就行 使用官方的更好

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][
                :, 2:4
            ] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(
                box_xy, box_wh, input_shape, image_shape, letterbox_image
            )
    return output
