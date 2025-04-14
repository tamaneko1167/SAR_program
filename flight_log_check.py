import sys
import numpy as np
import SA_func as sa

dir_name = "../send/2024-06-20/06-20-16-44-49/"
log_filename = dir_name + "flight_log"

args = sys.argv
window_index = int(args[1])

data = np.genfromtxt(log_filename + ".csv", delimiter=',', encoding='utf_8_sig')  # 区切り文字としてカンマを指定
xy_speed = np.sqrt(data[:, 19] ** 2 + data[:, 20] ** 2) * 1.60934 * 1000 / 3600
v_lis = xy_speed[window_index : window_index + int(sa.az_n / 10)]
#xyz_speed = np.sqrt(data[:, 19] ** 2 + data[:, 20] ** 2+ data[:, 21] ** 2) * 1.60934 * 1000 / 3600
#v_lis = xyz_speed[window_index : window_index + int(sa.az_n / 10)]
#v_lis = xy_speed[window_index :]
print(sum(v_lis * 0.1))
np.save(dir_name + "flight_log_v", v_lis)
#np.save(dir_name + "flight_log_v", xy_speed)