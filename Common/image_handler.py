# _*_ coding: utf-8 _*_
import numpy as np
import cv2
import os
import time
import itertools
from threading import Thread
from numpy.linalg import norm
from matplotlib import pyplot as plt
from log import Log


def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    output = itertools.zip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    """Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    """
    imgs = iter(imgs)
    img0 = next(imgs)
    pad = np.zeros_like(img0)
    imgs = itertools.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


class Assistant:
    def __init__(self):
        self.__log = Log()

    def auto_rename_files_in_dir(self, to_process_flag, exists_flag, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        while True:
            filenames = [i for i in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, i))]
            for filename in filenames:
                if exists_flag not in filename:
                    index = filename.find(to_process_flag)
                    if index == -1:
                        char, ext = os.path.splitext(filename)
                        alike = [i for i in filenames if i[:i.find(exists_flag)] == char]
                    else:
                        char = filename[:index]
                        alike = [i for i in filenames if i[:i.find(exists_flag)] == char]
                        ext = os.path.splitext(filename)[1]
                    num = len(alike) + 1
                    while True:
                        des_name = '%s%s%s%s' % (char, exists_flag, num, ext)
                        try:
                            os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, des_name))
                            break
                        except:
                            num += 1
                    self.__log.show_log(u'已将 %s 重命名为：%s' % (filename, des_name))
            time.sleep(0.3)

    def make_to_process(self, ext, check_flag, to_make_filename, uncared='', path='.'):
        """
        ext = '.png'
        uncared = '1_processed.png'
        check_flag = '-'
        to_make_filename = '1.png'
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        while True:
            filenames = [i for i in os.listdir(path) if os.path.splitext(i)[1] == ext and i != uncared]
            for filename in filenames:
                if check_flag in filename:
                    break
            else:
                if filenames:
                    try:
                        os.rename(os.path.join(path, filenames[0]), os.path.join(path, to_make_filename))
                        # 做开运算
                        # self.img_file_opening(os.path.join(path, to_make_filename))
                        self.__log.show_log(u'已生成%s' % to_make_filename)
                    except:
                        self.__log.show_log(u'致命错误！%s已存在' % to_make_filename)
            time.sleep(0.3)


class ImageHandler:
    """
    use opencv
    """
    @staticmethod
    def imread(file_full_path, flag=1):
        img = cv2.imdecode(np.fromfile(file_full_path, dtype=np.uint8), flag)
        return img

    @staticmethod
    def imwrite(self, file_full_path, img):
        file_ext = os.path.splitext(file_full_path)[1]
        cv2.imencode(file_ext, img)[1].tofile(file_full_path)

    @staticmethod
    def get_gray_img(file_full_path=None, img=None):
        """
        file_full_path, img 必须至少给出其一，否则返回 None.
        """
        if img is None:
            if file_full_path is None:
                return None
            img = imread(file_full_path, flag=1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img
    
    @staticmethod
    def get_threshold_img(self, file_full_path=None, img=None, gray_img=None, threshold_value=None, inverse=False):
        """
        file_full_path, img, gray_img 必须至少给出其一，否则返回 None.
        """
        if gray_img is None:
            if img is None:
                if file_full_path is None:
                    return None
                img = self.imread(file_full_path, flag=1)
            gray_img = self.get_gray_img(img=img)
        if threshold_value is None:
            threshold_value = (0, 255)
        _, threshold_img = cv2.threshold(gray_img, threshold_value[0], threshold_value[1], cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY)
        return threshold_img

    @staticmethod
    def get_translation_img(self, des_img=None, file_full_path=None, img=None, gray_img=None, threshold_value=None, inverse=False):
        """
        des_img: 'img' or 'gray_img', 'threshold_img'
        file_full_path, img, gray_img 必须至少给出之一，否则返回 None.
        """
        if des_img is None:
            return None

        if gray_img is None:
            if img is None:
                if file_full_path is None:
                    return None
                img = self.imread(file_full_path, flag=1)
                if des_img == 'img':
                    return img
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if des_img == 'gray_img':
                return gray_img
        if threshold_value is None:
            threshold_value = (0, 255)
        _, threshold_img = cv2.threshold(gray_img, threshold_value[0], threshold_value[1], cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY)
        if des_img == 'threshold_img':
            return threshold_img

    @staticmethod
    def cut_subrange(self, source_file_full_path, des_file_full_path, x, y, w, h):
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        img = self.imread(source_file_full_path)
        sub_img = img[y:y+h, x:x+w]
        self.imwrite(des_file_full_path, sub_img)

    #@staticmethod
    def find_threshold(self, file_full_path=None, img=None, gray_img=None, kernel_value=1):
        """
        file_full_path, img, gray_img 必须至少给出其一，否则返回 None.
        """
        # check params
        if gray_img is None:
            if img is None:
                if file_full_path is None:
                    return None
                img = self.imread(file_full_path, flag=1)
            gray_img = self.get_gray_img(img=img)

        # window
        cv2.namedWindow('Find threshold', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('threshold value', 'Find threshold', 0, 255, lambda x: None)

        # find value
        kernel = np.ones((kernel_value, kernel_value), np.uint8)
        while True:
            threshold_value = cv2.getTrackbarPos('threshold value', 'Find threshold')
            _, threshold_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
            opening_img = cv2.morphologyEx(threshold_img, op=cv2.MORPH_OPEN, kernel=kernel)
            cv2.imshow('Find threshold', opening_img)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord(' '):
                break

    @ staticmethod
    def get_irregular_roi(self, img, contour):
        roi_mask_inv = img
        roi_mask = cv2.bitwise_not(roi_mask_inv)
        roi_bg = cv2.bitwise_and(roi, roi, mask=roi_mask)

    def canny(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('min_value', 'canny', 0, 1000, lambda x: None)
        cv2.createTrackbar('max_value', 'canny', 0, 1000, lambda x: None)

        while True:
            min = cv2.getTrackbarPos('min_value', 'canny')
            max = cv2.getTrackbarPos('max_value', 'canny')
            edges = cv2.Canny(gray_img, min, max)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            cv2.imshow('canny1', edges)
        cv2.destroyAllWindows()

    def erode(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('kernel_value', 'erode', 0, 10, lambda x: None)

        while True:
            kernel_value = cv2.getTrackbarPos('kernel_value', 'erode')
            kernel = np.ones((kernel_value, kernel_value), np.uint8)
            erosion = cv2.erode(threshold_img, kernel, iterations=1)

            cv2.imshow('src', threshold_img)
            cv2.imshow('img', erosion)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        cv2.destroyAllWindows()

    def test_openning_kernel_value(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('kernel_value', 'opening', 0, 10, lambda x: None)

        while True:
            kernel_value = cv2.getTrackbarPos('kernel_value', 'opening')
            kernel = np.ones((kernel_value, kernel_value), np.uint8)
            opening_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

            cv2.imshow('opening1', opening_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_opening_img(self, file_full_path=None, img=None, gray_img=None, threshold_img=None, kernel_value=3, saved_path=None):
        """
        file_full_path, img, gray_img, threshold_img 必须至少给出其一，否则返回 None.
        """
        if threshold_img is None:
            if gray_img is None:
                if img is None:
                    if file_full_path is None:
                        return None
                    img = self.imread(file_full_path, flag=1)
                gray_img = self.get_gray_img(img)
            threshold_img = self.get_threshold_img(gray_img, inverse=True)

        kernel = np.ones((kernel_value, kernel_value), np.uint8)
        opening_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        if saved_path is not None:
            self.imwrite(file_full_path=file_full_path, img=opening_img, file_ext=os.path.splitext(saved_path)[1])

        return opening_img
        
    def find_contour(self, img, is_bg_reverse=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #if is_bg_reverse:
            #threshold_img = np.vectorize(lambda x: 0 if x == 255 else 255, otypes=[np.ndarray])(src_threshold_img).astype(np.uint8)

        # 查找轮廓
        image, contours, hierarchy = cv2.findContours(threshold_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('contours: ', len(contours))
        contour = contours[0]
        M = cv2.moments(contour)
        print(M)

        # 轮廓近似
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        print('epsilon', epsilon)
        #print('approx', approx)

        # 轮廓矩形
        x, y, w, h = cv2.boundingRect(contours[3])
        img1 = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        print('x, y, w, h: ', x, y, w, h)

        # 旋转的轮廓矩形
        box2d = cv2.minAreaRect(contours[3])
        print('box2D: ', box2d)

        #cx = int(M['mu10'] / M['mu00'])
        #cy = int(M['mu01'] / M['mu00'])
        #print(cx)
        #print(cy)
        rect_imgs = []
        print(len(contours))
        for i in range(len(contours)):
            rect_img = cv2.drawContours(img.copy(), contours, i, (0, 255, 0), 1)
            rect_imgs.append(rect_img)
            plt.subplot(2, len(contours), len(contours)+1+i), plt.imshow(rect_img), plt.title('rect_image%s' % (i+1)), plt.xticks([]), plt.yticks([])

        # 查看结果
        plt.gcf().set_size_inches(12, 2.2)
        plt.subplot(200+len(contours)*10+1), plt.imshow(self.get_rgb_img_from_bgr_img(bgr_img=img)), plt.title('img'), plt.xticks([]), plt.yticks([])
        plt.subplot(200+len(contours)*10+2), plt.imshow(gray_img, 'gray'), plt.title('gray_img'), plt.xticks([]), plt.yticks([])
        plt.subplot(200+len(contours)*10+3), plt.imshow(threshold_img, 'gray'), plt.title('threshold_img'), plt.xticks([]), plt.yticks([])

        #plt.subplot(234), plt.imshow(image), plt.title('image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(235), plt.imshow(img, 'gray'), plt.title('img'), plt.xticks([]), plt.yticks([])
        #plt.subplot(236), plt.imshow(rect_img), plt.title('rect_img'), plt.xticks([]), plt.yticks([])
        plt.show()


    def find_contours(self, img, is_bg_reverse=False):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 175, 255, cv2.THRESH_BINARY_INV)
        

    def deskew(self, img, is_bg_reverse=False, img_size=(0, 0)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if img_size == (0, 0):
            img_size = (max(gray_img.shape), max(gray_img.shape))

        # 二值化
        #_, threshold_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #if is_bg_reverse:
        #    threshold_img = np.vectorize(lambda x: 0 if x == 255 else 255, otypes=[np.ndarray])(threshold_img).astype(np.uint8)

        # 融合
        bg_img = np.repeat(0, img_size[0]*img_size[1]).reshape(img_size).astype(np.uint8)
        line_start = 0 if bg_img.shape[0] == img.shape[0] else (bg_img.shape[0] - img.shape[0]) // 2 + 1
        line_end = line_start + img.shape[0]
        column_start = 0 if bg_img.shape[1] == img.shape[1] else (bg_img.shape[1] - img.shape[1]) // 2 + 1
        column_end = column_start + img.shape[1]
        bg_img[line_start:line_end, column_start:column_end] = threshold_img

        # 去歪斜
        m = cv2.moments(bg_img)
        if abs(m['mu02']) < 1e-2:
            return bg_img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5*img_size[0]*skew], [0, 1, 0]])
        bg_img = cv2.warpAffine(bg_img, M, img_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return bg_img

    def match(self, file_full_path=None, img=None, template_file_full_path=None, template_img=None, method=None):
        """
        file_full_path     和 img          必须至少给出其一
        template_file_path 和 template_img 必须至少给出其一
        否则返回 None

        method 可不给

        正常返回 (img, template_top_left_location, template_width, template_height)
        """
        # img
        if img is None:
            if file_full_path is None:
                return None
            img = self.imread(file_full_path=file_full_path, flag=0)
        
        # template img
        if template_img is None:
            if template_file_full_path is None:
                return None
            template_img = self.imread(file_full_path=template_file_full_path, flag=0)
        
        w, h = template_img.shape[::-1]

        if method is None:
            method = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                      'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'][0]
        method = eval(method)

        result = cv2.matchTemplate(img.copy(), template_img, method)
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_location
        else:
            top_left = max_location
        bottom_right = (top_left[0]+w, top_left[1]+h)

        return (img, top_left, w, h)

    def get_color_img(self, file_full_path=None, img=None, color_name=None, color_hsv_threshold=None):
        """
        file_full_path 和 img                 必须至少给出其一,
        color_name     和 color_hsv_threshold 必须至少给出其一,
        否则返回 None.

        推荐给出 color_hsv_threshold, 会更准确。
        color_hsv_threshold: ([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v])

        H 范围: [0, 179]
        S 范围: [0, 255]
        V 范围: [0, 255]
        如果从其他软件得到的HSV值，注意转换。
        """
        if img is None:
            if file_full_path is None:
                return None
            img = self.imread(file_full_path, flag=1)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if color_hsv_threshold is None:
            if color_name is None:
                return None
            if color_name == '橙色':
                lower = [10, 50, 50]
                upper = [50, 255, 255]
            elif color_name == '紫色':  # photoshop: 270->135
                lower = [100, 100, 100]
                upper = [150, 255, 255]
            elif color_name == '绿色':  # 120->60, 100%->255, 50%->126
                lower = [20, 100, 80]
                upper = [100, 255, 255]
            elif color_name == '蓝色':  # 240->120, 100%->255, 100%->255
                lower = [90, 100, 100]
                upper = [150, 255, 255]
            elif color_name == '红色':  # 354->177, 94%->240, 79%->201 # 2->2, 88%->220, 80%->200
                lower = [160, 100, 80]
                upper = [190, 255, 255]
            color_hsv_threshold = (lower, upper)

        lower = np.array(color_hsv_threshold[0])
        upper = np.array(color_hsv_threshold[1])

        mask = cv2.inRange(hsv_img, lower, upper)
        return mask

    @staticmethod
    def get_rgb_img_from_bgr_img(self, bgr_img):
        b, g, r = cv2.split(bgr_img)
        return cv2.merge((r, g, b))

    @staticmethod
    def show_img(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class KNearestOcr(ImageHandler):
    def __init__(self):
        # 可识别字符
        self.__labels = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        # dir path
        self.__training_path = os.path.join(os.path.split(__file__)[0],'training_imgs')
        self.__success_path = 'success'
        self.__predict_path = 'predict_imgs'
        self.__threshold_value = 130

        # 数字到字符的映射
        self.__label_map = {}
        for i in range(len(self.__labels)):
            self.__label_map[int(i)] = self.__labels[i]

        # knn
        self.__k = 3
        self.__train_finished = False
        self.__knn = cv2.ml.KNearest_create()

        # log
        self.__log = Log()

    def test(self):
        filenames = [i for i in os.listdir(self.__predict_path) if os.path.isfile(os.path.join(self.__predict_path, i)) and i.split('.')[1] == 'jpg']
        for i in filenames:
            result = self.predict(os.path.join(self.__predict_path, i))
            self.__log.show_log(u'图片 %s 的识别结果为：%s' % (i, result))

    def get_img_text(self, img_path):
        result = self.split_img(file_path=img_path, parts_num=4)
        return result

    def start(self):
        assistant = Assistant()

        auto_reanme_success_thread = Thread(target=assistant.auto_rename_files_in_dir, args=(('-', '_', self.__success_path)))
        auto_reanme_training_thread = Thread(target=assistant.auto_rename_files_in_dir, args=(('-', '_', self.__training_path)))
        process_need_to_processing_thread = Thread(target=self.process_need_to_processing, args=(('1.png', '1_processed.png', '-')))
        auto_reanme_success_thread.start()
        auto_reanme_training_thread.start()
        process_need_to_processing_thread.start()
        assistant.make_to_process(ext='.png', check_flag='-', to_make_filename='1.png', uncared='1_processed.png')

    def process_need_to_processing(self, to_process_filename, processed_filename, split_imgs_flag):
        while True:
            filenames = [i for i in os.listdir() if os.path.isfile(i)]
            for filename in filenames:
                if filename == to_process_filename:
                    self.split_img(file_path=to_process_filename, parts_num=4, file_ext='.png', str_between_ocr_and_num=split_imgs_flag, need_clear_temp=False)
                    if os.path.isfile(processed_filename):
                        os.remove(processed_filename)
                    os.rename(filename, processed_filename)
                    break
            time.sleep(0.3)

    def predict(self, file_path):
        if not self.__train_finished:
            training_result = self.__train(self.__training_path, 'line')
            if not training_result:
                return 'no'

        # 预测
        predict_img = cv2.imread(file_path)
        predict_gray_img = cv2.cvtColor(predict_img, cv2.COLOR_BGR2GRAY)
        img_lines, img_columns = predict_gray_img.shape
        predict_data = predict_gray_img.reshape(-1, img_lines * img_columns).astype(np.float32)
        retval, result, neighbours, dists = self.__knn.findNearest(predict_data, k=self.__k)

        # 返回结果
        return self.__label_map[result[0, 0]]

    def split_img(self, file_path, parts_num, file_ext='.png', str_between_ocr_and_num='_', need_clear_temp=True, path='.', saved_path='.'):
        img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, self.__threshold_value, 255, cv2.THRESH_BINARY)
        #_, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_lines, img_columns = gray_img.shape
        columns = int(img_columns / parts_num)

        ocr_result = ''
        for i in range(parts_num):
            img_part = threshold_img[:, columns*i:columns*(i+1)]
            img_deskew = self.deskew(img_part)

            # save temp file
            filename = 'img%s.jpg' % i
            cv2.imwrite(filename, img_deskew)

            # rename
            img_str = self.predict(filename)
            ocr_result += img_str
            #shutil.move(filename, os.path.join(saved_path, filename))
            if need_clear_temp:
                os.remove(os.path.join(path, filename))
            else:
                ext_num = 0
                while True:
                    try:
                        os.rename(os.path.join(path, filename), os.path.join(saved_path, '%s%s%d%s' % (img_str, str_between_ocr_and_num, ext_num, file_ext)))
                        break
                    except:
                        ext_num += 1
        self.__log.show_log(u'==============================>识别结果为：%s' % ocr_result)
        return ocr_result

    def __train(self, training_dir, mode='line'):
        self.__log.show_log(u'训练数据......')

        # 读取训练图片
        # training_imgs = [
        #     ['3.jpg', '3_1.jpg', '3_2.jpg'],
        #     ['q.jpg', 'q_1.jpg', 'q_2.jpg']
        # ]
        # training_imgs = ['1.jpg', '2.jpg', '1.jpg']
        training_imgs = []
        if mode == 'line':
            training_imgs = os.listdir(training_dir)
            if not training_imgs:
                return False
            training_labels = [self.__get_label_int(i[0]) for i in training_imgs]
        elif mode == 'grid':
            filenames = os.listdir(self.__training_path)
            for filename in filenames:
                for line in training_imgs:
                    if filename[0] == line[0][0]:
                        line.append(filename)
                        break
                else:
                    training_imgs.append([filename])
            training_labels = [self.__get_label_int(i[0][0]) for i in training_imgs]

        # 加载训练图像
        if mode == 'line':
            imgs = [cv2.imread(os.path.join(training_dir, i)) for i in training_imgs]
            gray_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
            img_lines, img_columns = gray_imgs[0].shape
        elif mode == 'grid':
            imgs = list(map(lambda line: list(map(lambda fn: cv2.imread(os.path.join(training_dir, fn)), line)), training_imgs))
            gray_imgs = list(map(lambda line: list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), line)), imgs))
            #imgs1 = np.vectorize(lambda filename: cv2.imread(os.path.join(training_dir, filename)), otypes=[np.ndarray])(training_imgs)
            #gray_imgs1 = np.vectorize(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), otypes=[np.ndarray])(imgs1)
            img_lines, img_columns = gray_imgs[0][0].shape

        # 分割训练图像：5000 cells each 20x20
        if mode == 'line':
            lines = []
            for i in gray_imgs:
                cells = [np.hsplit(row, 1) for row in np.vsplit(i, 1)]
                cells = np.array(cells)
                linea = np.hstack(cells[:])
                lines.append(linea)
            cells = np.vstack(tuple(lines))
        elif mode == 'grid':
            lines = []
            for line in gray_imgs:
                cells = list(map(lambda x: [np.hsplit(row, 1) for row in np.vsplit(x, 1)], line))
                cells = list(map(np.array, cells))
                linea = np.hstack(cells[:])
                lines.append(linea)
            cells = np.vstack(tuple(lines))

        # 训练数据 和 测试数据
        train_data = cells.reshape(-1, img_lines * img_columns).astype(np.float32)      # shape(2, 400)
        test_data = cells.reshape(-1, img_lines * img_columns).astype(np.float32)   # shape(2, 400)

        # 标签
        if mode == 'line':
            train_labels = np.repeat(training_labels, 1)[:, np.newaxis]         # shape(2500, 1)
            test_labels = train_labels.copy()
        elif mode == 'grid':
            train_labels = np.repeat(training_labels, cells.shape[1])[:, np.newaxis]         # shape(2500, 1)
            test_labels = train_labels.copy()

        # 训练
        self.__knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

        retval, result, neighbours, dists = self.__knn.findNearest(test_data, k=self.__k)
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        self.__log.show_log(u'预测正确率为：%s%%' % accuracy)

        self.__train_finished = True
        return True

    def __get_label_int(self, char):
        for key, value in self.__label_map.items():
            if value == char.upper():
                return key


class SVMOcr(ImageHandler):
    def __init__(self, C=1, gamma=0.5):
        self.__img_size = (20, 20)
        self.__labels_chars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.__img_path = 'digits.png'
        self.__model = cv2.ml.SVM_create()
        self.__model.setGamma(gamma)
        self.__model.setC(C)
        self.__model.setKernel(cv2.ml.SVM_RBF)
        self.__model.setType(cv2.ml.SVM_C_SVC)
        self.__log = Log()

    def start(self):
        digits, labels = self.__load_digits(self.__img_path)

        self.__log.show_log('preprocessing......')
        # shuffle digits
        rand = np.random.RandomState(321)
        self.__log.show_log(rand)
        shuffle = rand.permutation(len(digits))
        digits, labels = digits[shuffle], labels[shuffle]

        self.__log.show_log('deskew......')
        digits2 = list(map(self.__deskew, digits))
        self.__log.show_log('hog......')
        samples = self.__preprocess_hog(digits2)

        train_n = int(0.9 * len(samples))
        cv2.imshow('test set', mosaic(25, digits[train_n:]))
        digits_train, digits_test = np.split(digits2, [train_n])
        samples_train, samples_test = np.split(samples, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])

        # train
        self.__log.show_log('train SVM......')
        self.__model.train(samples_train, labels_train)
        vis = self.__evaluate_model(digits_test, samples_test, labels_test)
        cv2.imshow('SVM test', vis)
        self.__log.show_log('saving SVM as "digits_svm.dat"......')
        self.__model.save('digits_svm.dat')

    def __split2d(self, img, cell_size, flatten=True):
        self.__log.show_log('split2d......')
        self.__log.show_log(img.shape)
        lines, columns = img.shape[:2]
        sx, sy = cell_size
        cells = [np.hsplit(row, columns//sx) for row in np.vsplit(img, lines//sy)]
        cells = np.array(cells)
        if flatten:
            cells = cells.reshape(-1, sy, sx)
        return cells

    def __load_digits(self, file_path):
        self.__log.show_log('loading "%s" ......' % file_path)
        digits_img = cv2.imread(file_path, 0)
        digits = self.__split2d(digits_img, self.__img_size)
        self.__log.show_log('create labels......')
        labels = np.repeat(self.__labels_chars, len(digits)/len(self.__labels_chars))
        return digits, labels

    def __deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5*self.__img_size[0]*skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, self.__img_size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def __evaluate_model(self, digits, samples, labels):
        self.__log.show_log(u'评估模型......')
        resp = self.__model.predict(samples)
        err = (labels != resp).mean()
        self.__log.show_log('error: %.2f %%' % (err * 100))

        '''
        confusion = np.zeros((10, 10), np.int32)
        for i, j in zip(labels, resp):
            confusion[i, int(j)] += 1
        self.__log.show_log('confusion matrix;')
        self.__log.show_log(confusion)
        '''

        vis = []
        for img, flag, in zip(digits, resp == labels):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if not flag:
                img[...,:2] = 0
            vis.append(img)
        return mosaic(25, vis)

    def __preprocess_hog(self, digits):
        samples = []
        for img in digits:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bin_n = 16
            bin = np.int32(bin_n * ang / (2 * np.pi))
            bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
            mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)


if __name__ == '__main__':
    #knn = KNearestOcr()
    #knn.start()
    #KNearestOcr().find_threshold('lk.jpg')
    #svm = SVMOcr()
    #svm.start()
    #img = cv2.imread('1_processed.png')
    #ocr = Ocr()
    #ocr.find_contour(img, True)
    #ocr.canny(img)
    #ocr.erode(img)
    #ocr.openning(img)
    img_handler = ImageHandler()
    #img_handler.find_threshold(r'F:\Projects\PythonProjects\Apps\PingAnDataNew\cut00083.png')
    img_handler.find_threshold('test1.png', kernel_value=2)
