# 未知尺寸圆柱体的只读视觉测量

更新：2026-10-21。目标外观由现场指定为红色侧面、白色端面；本模块使用红色侧面的 RGB-D 点云拟合圆柱曲面，不输入直径、高度或直立方向。白色端面目前由现场图像确认，程序尚未独立验证两个白色端面，也不输出完整物体高度或几何中心。

## 运行

在项目根目录编译后，每个终端加载 ROS 环境。相机运行中执行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run piperx_lab cylinder_target --samples 5 --timeout 25 \
  --out records/cylinder_real/observation.json \
  --save-frame records/cylinder_real/recognized.png
```

程序只读相机数据。采集完成后退出，输出 JSON 和可选标注图。`records/cylinder_real` 按项目规则保留在本地，不进入 Git。

## 流水线与字段

对齐彩色图、深度与内参按同一时间戳组合，使用当前 C1W 鱼眼参数反投影。选择最大红色区域并去除边缘，以三个主成分方向初始化稳健圆柱拟合，估计轴线和半径。可见曲率不足、区域近似平面、轴向跨度太短或拟合残差过大时拒绝结果。

输出相机系 `camera_depth_frame`，米制：

- `diameter_m`：根据可见曲面拟合的直径。
- `axis_direction`：单位轴向量，正负方向任意；绕轴旋转不能由圆形曲面确定。
- `axis_reference_xyz_m`：可见红色段在轴线上的中段参考点。它不等于整个物体中心，也不是夹爪 TCP 目标。
- `observed_red_span_m`：红色有效点沿轴向的 2%–98% 分位跨度。它是受遮挡、分割和腐蚀影响的可见跨度，不是完整高度。
- `surface_rmse_m`、`visible_arc_deg`、`inlier_fraction`：拟合一致性指标，不能当作实物测量误差或统计置信度。
- `full_height_m`、`object_center_xyz_m`：当前保持 `null`，不推测不可见部分。

ROS 只读输出 `/piperx/cylinder_axis_reference_camera`（PointStamped）与 `/piperx/cylinder_axis_camera`（Vector3Stamped）。这些输出不会触发机械臂规划或运动。

官方 Vidu wrapper 为深度对齐彩色图使用 `camera_aligned_color_frame` 标签。本模块及方块模块仅为该已知对齐流接受这个别名；深度与内参仍必须属于 `camera_depth_frame`，原始 `camera_color_frame` 不会被当作对齐输入。

## 本次实测与剩余工作

2026-10-21 在当前 C1W 和现场目标上连续记录了五帧有效拟合，直径均值约 43.1 mm、帧间标准差约 0.41 mm，可见红色轴向跨度均值约 42.5 mm。未使用尺量真值，这些数据只证明实时输入和拟合已运行，不代表绝对尺寸精度。

下一步需要测量工具验证直径，分离并拟合白色端面或补充视角确定完整高度与中心，再用观测时刻手眼变换得到基座系轴线，结合夹爪模型、TCP 和碰撞检查生成抓取候选。目前没有执行圆柱抓取。
