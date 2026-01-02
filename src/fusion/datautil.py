import math

import cv2
import numpy as np
from PIL import Image


class ProcessingKeypoints():
    def __init__(self):
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                        [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], \
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                       [85, 0, 255], \
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def trans_keypoins(self, keypoints, img_size, param):
        missing_keypoint_index = keypoints == -1

        # crop the white line in the original dataset
        keypoints[:, 0] = (keypoints[:, 0] - param['offset'])

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0 / param['anno_width'] * img_w
        scale_h = 1.0 / param['anno_height'] * img_h

        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
        else:
            w, h = 0, 0

        keypoints[:, 0] = keypoints[:, 0] * scale_w - w
        keypoints[:, 1] = keypoints[:, 1] * scale_h - h
        keypoints[missing_keypoint_index] = -1
        return keypoints
    def draw_img(self, keypoint, img_size, param):
        canvas = np.zeros((img_size[0], img_size[1], 3)).astype(np.uint8)
        stickwidth = param['stickwidth']
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(self.limbSeq[i]) - 1, 0]
            X = keypoint[np.array(self.limbSeq[i]) - 1, 1]
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        label_tensor = pose
        return label_tensor

    def get_label_tensor(self, path, img, param):
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, img.shape[:2], param)
        label_tensor = self.draw_img(keypoint, img.shape[:2], param)
        return label_tensor


class ProcessingSignKeypoints():
    def __init__(self):
        self.limbSeq = [[112, 130], [112, 126], [112, 122], [112, 118], [112, 114], [91, 93], [91, 97], [91, 101],
                        [91, 105], [91, 109],
                        [130, 132], [126, 128], [122, 124], [118, 120], [114, 116], [93, 95], [97, 99], [101, 103],
                        [105, 107]]

        a = [[11,9],[9,7],[7,6],[6,8],[8,10],[7,13],[6,12],[13,12],[13,15],
         [15,17],[12,14],[14,16],[24,25],[25,26],[26,27]] #[5,3],[3,1],[1,2],[2,4],

        f1 = [[i, i+1] for i in range(24,40)]
        f2 = [[i, i+1] for i in range(41,50)]
        f3 = [[i, i+1] for i in range(51,59)] + [[59,54]]
        f4 = [[i, i+1] for i in range(60,65)] + [[65,60]]
        f5 = [[i, i+1] for i in range(66,71)] + [[71,66]]
        f6 = [[i, i+1] for i in range(72,88)] + [[83,72] + [72,84] + [88,78]]
        h1 = [[96,95],[95,94],[94,93],[93,92],[100,99],[99,98],[98,97],[97,92],[104,103],[103,102],
         [102,101],[101,92],[108,107],[107,106],[106,105],[105,92],[112,111],[111,110],[110,109],
         [92,109]]
        h2 = [[133,132],[131,130],[130,113],[129,128],[128,127],[127,126],[126,113],[125,124],
         [124,123],[123,122],[122,113],[121,120],[120,119],[119,118],[118,113],[117,116],
         [116,115],[115,114],[114,113]]

        self.limbSeq = a + f1 + f2 + f3+ f4+f5+f6+h1+h2

        color_d = np.arange(0, 256, 1)
        np.random.seed(777)
        color_c = True
        if color_c == True:
            colors_list = np.random.choice(color_d, 3 * 133).reshape(133, 3)
            self.colors = colors_list.tolist()
        else:
            self.colors = [[125, 125, 125]]

    def trans_keypoins(self, keypoints, param, img_size):
        missing_keypoint_index = keypoints == 0
        keypoints[:, 0] = (keypoints[:, 0] - param['offset'])
        img_h, img_w = img_size
        scale_w = 1.0 / param['anno_width'] * img_w
        scale_h = 1.0 / param['anno_height'] * img_h


        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
            w, h = 0, 0  ## 주의
        else:
            w, h = 0, 0

        keypoints[:, 0] = keypoints[:, 0] * scale_w - w
        keypoints[:, 1] = keypoints[:, 1] * scale_h - h
        keypoints[missing_keypoint_index] = -1
        return keypoints

    def get_label_tensor(self, path, img, param):
        canvas = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
        # keypoint = np.loadtxt(path)
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, param, img.shape[:2])
        stickwidth = param['stickwidth']
        for i in range(len(keypoint)):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 2, self.colors[i], thickness=-1)
        joints = []
        for i in range(117):
            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        label_tensor = pose  # torch.cat((pose, tensors_dist), dim=0)
        return label_tensor
