"""
Description  : 
Version      : 1.0.0
Author       : Xu Huasheng
Date         : 2024-06-17 14:46:47
LastEditTime : 2024-08-21 10:12:59
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
                  save_faild: bool = True) -> List[str]:
    """识别DM码/QR码/字符(仅限数字和字母).
    Args:
        - img (np.ndarray): 包含码区/字符的图像.
        - type (str): "DM" | "QR" | "CHAR"
        - rotation (int): 识别非正置字符需要此参数, 
            `img`旋转至字符为正所需的角度(逆时针方向为正), 仅限0/90/180/270.
        - save_faild (bool): 是否保存识别失败的图片.
    Returns:
        - result (list): 码/字符, 无码或识别失败则返回[]
    """
    if type in ("DM", "QR"):
        if type == "DM":
            decoder = pylibdmtx
        else:
            decoder = pyzbar
        codes = decoder.decode(img)
        result = []
        if len(codes) != 0:
            result = [str(c.data, encoding='utf-8') for c in codes]
    else:
        import cnocr  # 导入时间较长 大约3s

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

    if save_faild and len(result) == 0:
        prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
        save_dir = os.path.join(LOG_DIR, f"/{prefix}")
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_code_failed.png"), img)
    return result


if __name__ == "__main__":
    img = cv2.imread("./test_img/DM.png")
    code = recogize_code(img, "DM")
    print(code)