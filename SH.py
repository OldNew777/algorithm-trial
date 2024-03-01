import cv2
import numpy as np
import math
import time
from scipy.special import sph_harm


class OctahedralMapping:
    @staticmethod
    def uv2xyz(uv: np.ndarray) -> np.ndarray:
        xy = uv * 2 - 1
        z = 1 - np.sum(np.abs(xy))
        if z < 0:
            xo = xy[0]
            xy[0] = 0.5 * (1 - np.abs(xy[1])) * np.sign(xo)
            xy[1] = 0.5 * (1 - np.abs(xo)) * np.sign(xy[1])
        xyz = np.array([xy[0], xy[1], z])
        xyz /= np.linalg.norm(xyz)
        return xyz

    @staticmethod
    def xyz2uv(xyz: np.ndarray) -> np.ndarray:
        xyz /= np.sum(np.abs(xyz))
        if xyz[2] >= 0:
            uv = xyz[:2]
        else:
            uv = np.array([
                (1 - np.abs(xyz[1])) * np.sign(xyz[0]),
                (1 - np.abs(xyz[0])) * np.sign(xyz[1])
            ])
        uv = (uv + 1) / 2
        return uv


def testOctahedralMapping():
    uv = np.array([0.6, 0.7])
    print(uv)
    xyz = OctahedralMapping.uv2xyz(uv)
    print(xyz)
    uv_remapped = OctahedralMapping.xyz2uv(xyz)
    print(uv_remapped)


def testSH():
    # timing
    time_start = time.time()

    # 读取图像
    image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
    image = 255 - image
    print(f'Image shape = {image.shape}, range = [{np.min(image)}, {np.max(image)}], average = {np.mean(image)}')

    # 图像转换为球面坐标系
    height, width = image.shape
    u_grid, v_grid = np.mgrid[0:1:complex(0, height), 0:1:complex(0, width)]
    # calculate xyz from uv as grid
    uv_grid = np.dstack((u_grid, v_grid))  # shape=(height, width, 2)

    # Mapping 1
    # apply OctahedralMapping.uv2xyz to uv_grid (this function accept shape=(2,) and return shape=(3,))
    xyz_grid = np.apply_along_axis(OctahedralMapping.uv2xyz, 2, uv_grid)
    # get theta, phi from xyz
    theta, phi = np.arccos(xyz_grid[..., 2]), np.arctan2(xyz_grid[..., 1], xyz_grid[..., 0])

    # Mapping 2
    # map uv directly to theta, phi
    theta, phi = np.pi * u_grid, 2 * np.pi * v_grid

    phi[phi < 0] += 2 * np.pi

    # 计算球谐函数
    num_sh = 4  # 球谐函数阶数
    sh_coeffs = np.zeros((num_sh, num_sh))
    for n in range(num_sh):
        for m in range(-n, n + 1):
            # 计算球谐函数系数
            k = math.sqrt((2 * n + 1) * math.factorial(n - m) / (4 * math.pi * math.factorial(n + m)))
            sh_coeffs[n, m] = np.sum(image * sph_harm(m, n, phi, theta).real) * k
    sh_coeffs /= (width * height)
    print(f'SH coefficients shape = {sh_coeffs.shape}, range = [{np.min(sh_coeffs)}, {np.max(sh_coeffs)}], average = {np.mean(sh_coeffs)}')

    # 重建图像
    reconstructed_image = np.zeros_like(image, dtype=np.float64)
    for n in range(num_sh):
        for m in range(-n, n + 1):
            # 重建图像
            reconstructed_image += sh_coeffs[n, m] * sph_harm(m, n, phi, theta).real

    # 将重建图像限制在0到255之间
    # reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    print(
        f'Reconstructed image shape = {reconstructed_image.shape}, '
        f'range = [{np.min(reconstructed_image)}, {np.max(reconstructed_image)}], average = {np.mean(reconstructed_image)}')

    time_end = time.time()
    print('Time cost =', time_end - time_start, 's')

    # 显示原始图像和重建图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Reconstructed Image', reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # testOctahedralMapping()
    testSH()
