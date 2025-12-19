import cv2
import numpy as np

def compute_colorfulness(image):
    """
    计算输入图像的颜色丰富度（Colorfulness）。
    参数：
        image: numpy.ndarray，BGR 格式的图像（OpenCV 读取后一般默认为 BGR）。
    返回：
        colorfulness: float，图像的颜色丰富度数值，值越大表示颜色越丰富。
    """
    # 1. 将图像转换为 float 类型，避免后续计算时数据溢出
    img = image.astype(np.float32)

    # 2. 分离 B, G, R 通道（OpenCV 读取的顺序为 BGR）
    (R, G, B) = cv2.split(img)

    # 3. 计算 R-G 分量（RG）和 (R+G)/2 - B 分量（YB）
    RG = np.abs(R - G)
    YB = np.abs(0.5 * (R + G) - B)

    # 4. 计算 RG 和 YB 的均值与标准差
    sigma_RG = np.std(RG)
    sigma_YB = np.std(YB)
    mean_RG = np.mean(RG)
    mean_YB = np.mean(YB)

    # 5. 代入公式：sqrt(sigma_RG^2 + sigma_YB^2) + 0.3 * sqrt(mean_RG^2 + mean_YB^2)
    std_root = np.sqrt(sigma_RG**2 + sigma_YB**2)
    mean_root = np.sqrt(mean_RG**2 + mean_YB**2)
    colorfulness = std_root + 0.3 * mean_root

    return colorfulness

if __name__ == "__main__":
    # 示例：读取一张图计算颜色丰富度
    img_path = r"D:\A_zhangxin\image_metric\sea\first\img832.png" # 替换为你自己的图片路径
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片，请检查路径是否正确。")
    else:
        c = compute_colorfulness(img)
        print(f"图像的颜色丰富度 = {c:.2f}")

