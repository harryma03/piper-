# RGB-D 目标点云观察

只读网页显示对齐彩色图、颜色掩码和相机光学坐标系点云。白色十字和数值是
颜色掩码内有效深度点的坐标中位数，代表可见表面，不是物体中心或抓取位姿。
绿色图像区域和黄色点云可以来自颜色分割，也可以选择“点击图像选目标”
并点击目标内部，以深度连通区域选取不同颜色的物体。深度连通会检查相邻像素
深度跳变，RGB 颜色不参与点选计算。深度相近、贴在一起的物体或桌面仍可能
误合并；透明、反光或深度缺失区域可能无法选中，必须目视检查。
白色端面不自动包含在红色掩码内。

先启动相应的相机数据源，再从项目根目录运行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=183
export ROS_LOCALHOST_ONLY=1
python3 scripts/pointcloud_viewer.py --source gazebo
```

浏览器访问 `http://127.0.0.1:8765`。Gazebo 页面使用仿真深度图与内参
反投影，只订阅传感器话题。鼠标拖动点云旋转，滚轮缩放；可选择仅目标、
场景与目标或仅场景。实时点选时，图像和点云使用同一帧并锁定供检查；点击
“刷新”重新获取实时帧并点选。相机或物体移动后必须重新选择。其他模式的数据
停更超过三秒时界面提示。Python 环境需要 Flask、NumPy、OpenCV 和 ROS 2。

查看 C1W 时，在相机 wrapper 已运行、ROS 域与相机节点一致的终端中启动：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/pointcloud_viewer.py --source real
```

实机模式订阅对齐图像、深度、相机内参和 SDK `/camera/pointcloud`，
要求同一时间戳，并检查 SDK 点云和有效深度像素数量匹配。该路径尚需
连接当前相机后现场验证。服务只监听本机地址，不发送机械臂运动命令。

没有实时相机时使用 `--source file`，并在页面的“数据来源”中选择历史
帧；包含 xyz、color、mask 的历史 NPZ 可选“历史帧原始掩码”。按 Ctrl+C
停止服务。多个服务同时运行时可通过 `--port` 指定不同端口。

仿真抓取实验支持手动像素种子：重启已更新的仿真感知服务后，
`run_closed_loop.py --mode single --target-pixel X,Y ...` 会按当前图像中的
该像素提取深度连通区域，然后交给 GraspNet；像素可从网页“点选像素”查看。
该入口仍是实验性仿真功能，尚未证明能可靠抓取非红色物体；移动相机时固定
像素不再指向同一物体，因此暂不支持多视角或分段跟踪模式。

## YOLO 物体实例识别

界面“识别当前画面”运行 YOLO11n-seg，选择识别出的实例后显示该实例对应的同帧 RGB-D 可见表面点云及其中位坐标。实时识别锁定当前帧；点“刷新”恢复观察。它是二维类别和实例掩码，不是六自由度位姿、物体几何中心或抓取指令。默认 COCO 预训练类别不涵盖所有物体，自制圆柱不保证被识别；看不到目标时可用点击深度选点。

该实验使用项目现有 Python 3.11 `.graspnet-venv` 中的 PyTorch，YOLO 依赖单独放在已忽略的 `third_party/yolo_runtime`，权重放在 `third_party/yolo_models`，不会自动提交。环境重建示例：

```bash
.graspnet-venv/bin/python -m pip install --target third_party/yolo_runtime --no-deps 'ultralytics==8.4.22' polars polars-runtime-32 ultralytics-thop
mkdir -p third_party/yolo_models
curl -fL -o third_party/yolo_models/yolo11n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
```

该命令依赖已有的 `.graspnet-venv` 及其 PyTorch、OpenCV 等依赖；换机器需先配置该环境。模型由 Ultralytics 提供，使用和分发前请核查其 [AGPL-3.0 / 企业许可](https://www.ultralytics.com/license)。
