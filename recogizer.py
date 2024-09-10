"""
Description  : 
Version      : 1.0.0
Author       : Xu Huasheng
Date         : 2024-06-17 14:46:47
LastEditTime : 2024-09-10 17:35:41
LastEditors  : aircas41-server-win xuhs@aircas.ac.cn
Copyright (c) 2024 AIRCAS. All rights reserved. 
"""
import os
import time
from typing import List, Literal

import cv2
import numpy as np
from pylibdmtx import pylibdmtx  # DM
from pyzbar import pyzbar  # QR

LOG_DIR = './'


def recogize_code(img: np.ndarray, 
                  type: Literal["DM", "QR", "CHAR"], 
                  rotation: int = 0, 
                  capture_roi: bool = True,
                  save_faild: bool = False) -> List[str]:
    """识别DM码/QR码/字符(仅限数字和字母).
    Args:
        - img (np.ndarray): 包含码区/字符的图像.
        - type (str): "DM" | "QR" | "CHAR"
        - rotation (int): 识别非正置字符需要此参数, 
            `img`旋转至字符为正所需的角度(逆时针方向为正), 仅限0/90/180/270.
        - capture_roi (bool): 是否抓取ROI.
        - save_faild (bool): 是否保存识别失败的图片.
    Returns:
        - result (list): 码/字符, 无码或识别失败则返回[]
    """
    if type in ("DM", "QR"):
        # 转灰度 缩放 二值化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        h_rate = h / 300
        w_rate = w / 300
        rate = max(h_rate, w_rate)
        img = cv2.resize(img, (int(w/rate), int(h/rate)))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 抓取ROI
        if capture_roi:
            t0 = time.perf_counter()
            contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            area_max = 0
            c_max = contours[0]
            for c in contours:
                area = cv2.contourArea(c)
                if area > area_max:
                    area_max = area
                    c_max = c
            x_min = min(c_max[:,0,0])
            x_max = max(c_max[:,0,0])
            y_min = min(c_max[:,0,1])
            y_max = max(c_max[:,0,1])
            img = img[y_min:y_max, x_min:x_max]
            print("抓取ROI", time.perf_counter() - t0)

        if type == "DM":
            t = time.perf_counter()
            codes = pylibdmtx.decode(img, timeout=1000)
            print("识别时间", time.perf_counter() - t)
            result = []
            if len(codes) != 0:
                result = [str(c.data, encoding='utf-8') for c in codes]
        else:
            codes = pyzbar.decode(img)
            result = []
            if len(codes) != 0:
                result = [str(c.data, encoding='utf-8') for c in codes]
    else:
        t0 = time.perf_counter()
        import cnocr  # 导入时间较长 大约3s
        print("导入", time.perf_counter()- t0)

        t0 = time.perf_counter()
        # 可选模型: https://cnocr.readthedocs.io/zh-cn/stable/models/
        ocr = cnocr.CnOcr(rec_model_name='en_number_mobile_v2.0')
        if rotation == 0:
            pass
        elif rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else: 
            raise ValueError("rotation must be 0/90/180/270")
        code = ocr.ocr(img)
        result = [c["text"] for c in code]
        print("识别", time.perf_counter()- t0)

    if save_faild and len(result) == 0:
        prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
        save_dir = os.path.join(LOG_DIR, f"/{prefix}")
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_code_failed.png"), img)
    return result


if __name__ == "__main__":
    img = cv2.imread(r"c:\Users\AIRCAS\Desktop\20240909174558_code_failed.jpg")
    import time
    t0 = time.perf_counter()
    code = recogize_code(img, "DM")
    print(code, time.perf_counter()- t0)