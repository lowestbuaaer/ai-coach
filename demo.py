import argparse
import math

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from PIL import Image, ImageDraw, ImageFont

# 第一个demo首先测试举杠铃动作，在动作不达标的时候使用红色来标识，然后红色逐渐变浅，变为浅绿色，然后绿色逐渐变深，在这个过程中圆环中的实心圆在不断地变大
class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

# 核心方法，在这个地方实现骨骼点坐标的输出
def run_demo(net, image_provider, height_size, cpu, track, smooth):
    fontpath = "./FZMingSTJW.TTF"  # 宋体字体文件
    font_1 = ImageFont.truetype(fontpath, 15)  # 加载字体, 字体大小
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            # 图片的尺寸为480*640
        if len(current_poses) > 0:
            mainpose = current_poses[0]
            # mainpose.keypoints 存储着拍摄到的主要人像的关节点的坐标信息，可以通过这个来计算动作的标准程度,下面这条语句可以用来画圆环或者实心圆，从而作为动作标准与否的实时可视化显示
            # 要去算四个角度，然后根据这个角度来判断动作的标准程度
            # 2 3 4 是右肩右肘右手腕   5 6 7 是左肩左肘左手腕  8  11是右胯和左胯
            if not (mainpose.keypoints[2][0] != -1 and mainpose.keypoints[2][1] != -1 and mainpose.keypoints[3][
                0] != -1 and mainpose.keypoints[3][1] != -1 and mainpose.keypoints[4][0] != -1 and
                    mainpose.keypoints[4][1] != -1 and mainpose.keypoints[5][0] != -1 and mainpose.keypoints[5][
                        1] != -1 and mainpose.keypoints[6][0] != -1 and mainpose.keypoints[6][1] != -1 and
                    mainpose.keypoints[7][0] != -1 and mainpose.keypoints[7][1] != -1):
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 100), "请退后几步，露出全身", font=font_1, fill=(0, 255, 0))  # xy坐标, 内容, 字体, 颜色
                img = np.array(img_pil)
            else:
                cv2.circle(img, (mainpose.keypoints[2][0], mainpose.keypoints[2][1]), 20, (209, 216, 129), 2)  # RGB是反着的
                cv2.circle(img, (mainpose.keypoints[3][0], mainpose.keypoints[3][1]), 20, (209, 216, 129), 2)  # RGB是反着的
                cv2.circle(img, (mainpose.keypoints[5][0], mainpose.keypoints[5][1]), 20, (209, 216, 129), 2)  # RGB是反着的
                cv2.circle(img, (mainpose.keypoints[6][0], mainpose.keypoints[6][1]), 20, (209, 216, 129), 2)  # RGB是反着的
                a = math.sqrt((mainpose.keypoints[2][0] - mainpose.keypoints[3][0]) ** 2 + (
                            mainpose.keypoints[2][1] - mainpose.keypoints[3][1]) ** 2)
                b = math.sqrt((mainpose.keypoints[2][0] - mainpose.keypoints[8][0]) ** 2 + (
                            mainpose.keypoints[2][1] - mainpose.keypoints[8][1]) ** 2)
                c = math.sqrt((mainpose.keypoints[8][0] - mainpose.keypoints[3][0]) ** 2 + (
                            mainpose.keypoints[8][1] - mainpose.keypoints[3][1]) ** 2)
                angle1 = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                # 右肩角度

                a = math.sqrt((mainpose.keypoints[3][0] - mainpose.keypoints[4][0]) ** 2 + (
                        mainpose.keypoints[3][1] - mainpose.keypoints[4][1]) ** 2)
                b = math.sqrt((mainpose.keypoints[3][0] - mainpose.keypoints[2][0]) ** 2 + (
                        mainpose.keypoints[3][1] - mainpose.keypoints[2][1]) ** 2)
                c = math.sqrt((mainpose.keypoints[2][0] - mainpose.keypoints[4][0]) ** 2 + (
                        mainpose.keypoints[2][1] - mainpose.keypoints[4][1]) ** 2)
                angle2 = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                # 右肘角度

                a = math.sqrt((mainpose.keypoints[5][0] - mainpose.keypoints[6][0]) ** 2 + (
                        mainpose.keypoints[5][1] - mainpose.keypoints[6][1]) ** 2)
                b = math.sqrt((mainpose.keypoints[5][0] - mainpose.keypoints[11][0]) ** 2 + (
                        mainpose.keypoints[5][1] - mainpose.keypoints[11][1]) ** 2)
                c = math.sqrt((mainpose.keypoints[11][0] - mainpose.keypoints[6][0]) ** 2 + (
                        mainpose.keypoints[11][1] - mainpose.keypoints[6][1]) ** 2)
                angle3 = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                # 左肩角度

                a = math.sqrt((mainpose.keypoints[6][0] - mainpose.keypoints[7][0]) ** 2 + (
                        mainpose.keypoints[6][1] - mainpose.keypoints[7][1]) ** 2)
                b = math.sqrt((mainpose.keypoints[6][0] - mainpose.keypoints[5][0]) ** 2 + (
                        mainpose.keypoints[6][1] - mainpose.keypoints[5][1]) ** 2)
                c = math.sqrt((mainpose.keypoints[5][0] - mainpose.keypoints[7][0]) ** 2 + (
                        mainpose.keypoints[5][1] - mainpose.keypoints[7][1]) ** 2)
                angle4 = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                # 左肘角度
                # 认为肩部角度越大越好  45度标准  认为肘部角度越小越好  90度标准
                # 按照角度在20的圆内画圆盘
                if angle1 > 0:
                    if angle1 < 3.14 / 8:
                        cv2.circle(img, (mainpose.keypoints[2][0], mainpose.keypoints[2][1]),
                                   int(20 * angle1 / 3.14 * 4), (0, 0, 255),
                                   -1)  # RGB是反着的
                    else:
                        if angle1 > 3.14 / 4:
                            angle1 = 3.14 / 4
                        cv2.circle(img, (mainpose.keypoints[2][0], mainpose.keypoints[2][1]),
                                   int(20 * angle1 / 3.14 * 4), (0, 255, 0),
                                   -1)  # RGB是反着的
                if angle3 > 0:
                    if angle3 < 3.14 / 8:
                        cv2.circle(img, (mainpose.keypoints[5][0], mainpose.keypoints[5][1]),
                                   int(20 * angle3 / 3.14 * 4), (0, 0, 255),
                                   -1)  # RGB是反着的
                    else:
                        if angle3 > 3.14 / 4:
                            angle3 = 3.14 / 4
                        cv2.circle(img, (mainpose.keypoints[5][0], mainpose.keypoints[5][1]),
                                   int(20 * angle3 / 3.14 * 4), (0, 255, 0),
                                   -1)  # RGB是反着的
                if angle2 <= 3.14:
                    if angle2 > 2.5:
                        loss = (3.14 - angle2) / 3.14 * 2
                        cv2.circle(img, (mainpose.keypoints[3][0], mainpose.keypoints[3][1]), int(20 * loss),
                                   (0, 0, 255),
                                   -1)  # RGB是反着的
                    else:
                        if angle2 < 3.14 / 2:
                            angle2 = 3.14 / 2
                        loss = (3.14 - angle2) / 3.14 * 2
                        cv2.circle(img, (mainpose.keypoints[3][0], mainpose.keypoints[3][1]), int(20 * loss),
                                   (0, 255, 0),
                                   -1)  # RGB是反着的
                if angle4 <= 3.14:
                    if angle4 > 2.5:
                        loss = (3.14 - angle4) / 3.14 * 2
                        cv2.circle(img, (mainpose.keypoints[6][0], mainpose.keypoints[6][1]), int(20 * loss),
                                   (0, 0, 255),
                                   -1)  # RGB是反着的
                    else:
                        if angle4 < 3.14 / 2:
                            angle4 = 3.14 / 2
                        loss = (3.14 - angle4) / 3.14 * 2
                        cv2.circle(img, (mainpose.keypoints[6][0], mainpose.keypoints[6][1]), int(20 * loss),
                                   (0, 255, 0),
                                   -1)  # RGB是反着的

                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                if angle1 < 3.14 / 5:
                    draw.text((100, 100), "请继续抬高右上臂", font=font_1, fill=(0, 255, 0))  # xy坐标, 内容, 字体, 颜色
                if angle3 < 3.14 / 5:
                    draw.text((100, 120), "请继续抬高左上臂", font=font_1, fill=(0, 255, 0))  # xy坐标, 内容, 字体, 颜色
                if angle2 > 2:
                    draw.text((100, 140), "请收紧右臂肱二头肌", font=font_1, fill=(0, 255, 0))  # xy坐标, 内容, 字体, 颜色
                if angle2 > 2:
                    draw.text((100, 160), "请收紧左臂肱二头肌", font=font_1, fill=(0, 255, 0))  # xy坐标, 内容, 字体, 颜色
                img = np.array(img_pil)


        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default=r'.\checkpoint_iter_370000.pth', help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')
    # 新建网络
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    # 把网络和权重重新结合
    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        # 新建好了一个视频读取类
    else:
        args.track = 0
# 实际的运行方法
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
