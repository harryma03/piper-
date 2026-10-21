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

下一步需要测量工具验证直径，分离并拟合白色端面或补充视角确定完整高度与中心，基座变换与规划预览已接通，见下节；实机执行前仍需验证接触 TCP 和完整碰撞场景。目前没有执行圆柱抓取。

## 基座坐标与抓取规划预览

相机、实机反馈和 MoveIt 已运行时，在项目根目录执行：

```bash
bash scripts/start_cylinder_preview.sh
```

脚本启动独立的 RViz 预览窗口、有限时长的观测节点和规划节点。Ctrl+C 结束本脚本的观测及窗口；不会停止已有机械臂驱动。该窗口不提供 Execute 操作。

观测节点的 `--base` 选项按观测时刻查找 `base_link → link6`，结合通过质量检查的手眼结果，将参考点和轴向量转换到基座系。精确 TF 不可用时，仅接受时间差不超过 0.2 秒的最新 TF，并记录回退情况。原子 JSON 话题 `/piperx/cylinder_observation` 同时携带相机和基座结果；另有 `/piperx/cylinder_axis_reference_base` 与 `/piperx/cylinder_axis_base`。

规划顺序为：

1. 从目标附近深度点拟合局部支撑平面；失败时阻止规划。
2. 生成 16 个轴向、径向和斜向夹持候选，夹爪闭合方向垂直圆柱轴线。
3. 使用官方夹爪模型的指尖中点作为临时夹持参考。该点距 link6 的模型轴向距离为 142.5 mm；运行时还读取 `link6 → tcp_link`，将指尖目标换算为 MoveIt 端点目标，避免把法兰直接放到物体处。
4. 对预抓取点求带碰撞检查的逆解，规划当前状态到预抓取点，再计算到夹持点的直线接近轨迹。无碰撞逆解仅用于失败诊断，不接受为执行路径。
5. 规划期间持续核对观测与实际关节是否仍新鲜，拒绝关节、目标位置、轴向或尺寸明显变化的快照。

夹爪开口在规划状态中假设为“估计直径 + 两侧各 10 mm 余量”；程序不会实际张开夹爪，也不调用执行接口。实际开口和假设开口分别写入记录。

结果保存在本地 `records/cylinder_real/grasp_preview.json`，包含输入观测、真实初始关节、候选、失败阶段、选中编号和完整规划关节轨迹。RViz 的专用轨迹话题为 `/piperx/cylinder_preview_trajectory`，标记话题为 `/piperx/cylinder_preview_markers`。这是一次观测快照的规划结果；执行前必须重新观测。

### 实测结果及边界

2026-10-21 的现场规划中，第 9 号斜向候选曾成功：预抓取轨迹 39 点，直线接近轨迹 15 点，接近比例 100%；该次假设夹爪开口约 61.6 mm，真实反馈约 5.7 mm。结果只证明服务链能生成规划，机械臂和夹爪均未由预览程序执行。

当前碰撞对象仅覆盖局部 0.4 × 0.4 m 桌面片和可见红色段；白色端面、完整物体外形及其他障碍物尚未完整建模。模型指尖中点也未经独立接触 TCP 测量。因此输出始终标记 `execution_ready: false`，不能据此直接执行抓取。

节点只添加或移除 `piper_cylinder_preview_support` 和 `piper_cylinder_preview_observed_band` 两个自有碰撞对象，保留其他场景对象。关闭预览后如需移除它们，加载 ROS 和工作区环境后执行：

```bash
ros2 run piperx_lab cylinder_preview --clear-scene
```
