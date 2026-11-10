# Piper-X 任意形状抓取：待解决事项

更新时间：2026-11-07

状态总览见 `docs/AGENT_HANDOFF_ARBITRARY_OBJECT_GRASP.md` §9/§10；实现细节见
`docs/GENERIC_OBJECT_OBSERVATION.md`。

## 一、被 GPU 阻塞（驱动 CUDA unknown error）

2026-11-07 上午 GPU 驱动进入故障状态：`nvidia-smi` 显示正常（显存充足、无残留进程），
但 CUDA 上下文创建失败（`torch.cuda.is_available() == False`，报 CUDA unknown error）。
用户态无法修复，需要重载 NVIDIA 驱动或重启机器。

### 1. 新形状试点重跑

分割与几何测量已验证正确（球 [36,32,40] mm、随机圆柱 [48,49,90]/[57,57,53]/[26,27,75] mm、
随机 box 含 ±180° 偏航测量准确），只差 GraspNet 推理和抓取统计。

```bash
cd /home/co/ROBOT_PIPERX
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape sphere --trials 3 --seed 911 \
  --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/generic_sphere_seed911
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape box --object-dims auto --object-yaw auto \
  --trials 3 --seed 912 --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/generic_randombox_seed912
python3 scripts/run_gazebo_graspnet_batch.py \
  --object-shape cylinder --object-dims auto \
  --trials 3 --seed 913 --top-models 4 --symmetry-samples 8 \
  --out records/graspnet_gazebo/generic_randomcyl_seed913
```

### 2. 圆柱 seed20261106 ×10 最终确认

现有 9/10 记录（`records/graspnet_gazebo/generic_random10_seed20261106/`）跑在
"凸包支撑面补全"修复之前；需在最终代码下复跑一次对齐统计口径。

```bash
rm -rf records/graspnet_gazebo/generic_random10_seed20261106
python3 scripts/run_gazebo_graspnet_batch.py \
  --trials 10 --seed 20261106 \
  --out records/graspnet_gazebo/generic_random10_seed20261106
```

### 3. 交接文档 §9 补全

把上述两个批次的最终统计写入 `docs/AGENT_HANDOFF_ARBITRARY_OBJECT_GRASP.md` §9
的"新形状试点"小节，替换当前的"待 GPU 恢复"占位描述。

三项合计约 1 小时仿真时间，按 1 → 2 → 3 顺序执行。

## 二、已知但未修的问题（非阻塞）

1. **新形状规划失败模式**：对称旋转把 1~2 个模型抓取复制成全部候选，一种坏的接近
   几何 = 全部候选同型失败；陡峭接近角（如球顶抓取）配合固定 10 cm 预抓取退距导致
   预抓取 IK 无解（-31）。已用 `--top-models 4` 缓解；根治方向：自适应预抓取距离、
   更多 ROI×模型组合、对对称物体保留多个不同深度变体。
2. **球类 OBB 水平尺寸偏小**（36 mm vs 真值 40 mm）：单视角可见冠层弦长小于赤道
   直径，是观测几何极限而非 bug；需要多视角融合才能消除。
3. **偏航 box 的 PCA-OBB 水平尺寸偏大**最多 ~5 mm：碰撞用凸包不受影响，仅
   `dimensions_m` 报告字段偏大。

## 三、后续路线（交接文档 §14 未完成项）

- 横放、倾斜、堆叠、遮挡场景。
- mesh（STL/OBJ）物体的完整抓取试点（`make_scene.py` 已支持 mesh，未跑抓取）。
- 实机 C1W 接入同一观测接口（SDK XYZ 路径，`build_generic_observation` 已留入口，
  实机节点未实现）。
- AnyGrasp 许可申请，或收集数据微调 GraspNet。
- 胶囊体形状（SDF 1.6 无原生胶囊，需改用 mesh 表示）。
- 仿真达到验收门槛后，设计实机低速验证（夹爪滑移、手眼漂移）。
