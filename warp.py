import cv2
import numpy as np

def B_spline(u, flag):
    if flag == 0:
        return (1 - u**3 + 3 * u**2 - 3 * u) / 6.0
    elif flag == 1:
        return (4 + 3 * u**3 - 6 * u**2) / 6.0
    elif flag == 2:
        return (1 - 3 * u**3 + 3 * u**2 + 3 * u) / 6.0
    elif flag == 3:
        return u**3 / 6.0
    else:
        return 0.0

def B_spline_form(srcimg, delta_x=32, delta_y=32):
    grid_rows = (srcimg.shape[0] // delta_x) + 1 + 3
    grid_cols = (srcimg.shape[1] // delta_y) + 1 + 3
    noiseMat = np.zeros((grid_rows, grid_cols, 2), dtype=np.float32)
    # uniform_distribution = random.uniform(-20, 20)

    for row in range(grid_rows):
        for col in range(grid_cols):
            noiseMat[row, col, 0] = np.random.uniform(-10, 10)
            noiseMat[row, col, 1] = np.random.uniform(-10, 10)

    dstimg = np.zeros_like(srcimg, dtype=np.uint8)
    offset = np.zeros((srcimg.shape[0], srcimg.shape[1], 2), dtype=np.float32)

    for x in range(srcimg.shape[0]):
        for y in range(srcimg.shape[1]):
            i = int(x / delta_x)
            j = int(y / delta_y)
            u = float(x / delta_x) - i
            v = float(y / delta_y) - j

            pX = [B_spline(u, k) for k in range(4)]
            pY = [B_spline(v, k) for k in range(4)]

            Tx, Ty = 0.0, 0.0
            for m in range(4):
                for n in range(4):
                    control_point_x = i + m
                    control_point_y = j + n
                    temp = pY[n] * pX[m]
                    Tx += temp * (noiseMat[control_point_x, control_point_y, 0])
                    Ty += temp * (noiseMat[control_point_x, control_point_y, 1])

            offset[x, y, 0] = Tx
            offset[x, y, 1] = Ty

    for row in range(dstimg.shape[0]):
        for col in range(dstimg.shape[1]):
            src_x = row + offset[row, col, 0]
            src_y = col + offset[row, col, 1]
            x1, y1 = int(src_x), int(src_y)
            x2, y2 = x1 + 1, y1 + 1

            if x1 < 0 or x1 > (srcimg.shape[0] - 2) or y1 < 0 or y1 > (srcimg.shape[1] - 2):
                dstimg[row, col] = [0, 0, 0]
            else:
                pointa = srcimg[x1, y1]
                pointb = srcimg[x2, y1]
                pointc = srcimg[x1, y2]
                pointd = srcimg[x2, y2]

                B = (x2 - src_x) * (y2 - src_y) * pointa[0] - (x1 - src_x) * (y2 - src_y) * pointb[0] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[0] + (x1 - src_x) * (y1 - src_y) * pointd[0]
                G = (x2 - src_x) * (y2 - src_y) * pointa[1] - (x1 - src_x) * (y2 - src_y) * pointb[1] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[1] + (x1 - src_x) * (y1 - src_y) * pointd[1]
                R = (x2 - src_x) * (y2 - src_y) * pointa[2] - (x1 - src_x) * (y2 - src_y) * pointb[2] - \
                    (x2 - src_x) * (y1 - src_y) * pointc[2] + (x1 - src_x) * (y1 - src_y) * pointd[2]

                dstimg[row, col] = [B, G, R]

    return dstimg