# Windows 安装与更新

## 已有这台电脑

安装程序会更新 BL62 Reconstruction，保留独立的 `tomocupy_astra` 和 `SAM3p1` 环境。
统一 checkpoint 默认仍为 `D:\Chcekpoints\checkpoint_demo_05142026.pt`，不复制或捆绑模型。
原版程序另有带时间戳的备份，可以使用备份目录中的 `run_gui.bat` 回退。

## 新电脑

1. 安装 NVIDIA 驱动、Miniforge。
2. 安装 Setup EXE，或解压 Portable ZIP。
3. 从开始菜单运行 **BL62 Reconstruction - Install Environments**。
   Portable 版运行 `_internal\install_windows.bat`。
4. 脚本创建缺少的重建和 SAM3 环境，并验证依赖；首次下载为数 GB。
5. 在 GUI 的 **Settings → SAM3** 选择 checkpoint。也可更改库和 Python 路径。

此发行版的自动分析在原生 Windows 运行，不依赖 WSL。
源代码安装时，在目录中运行 `install_windows.bat`；之后使用 `run_gui.bat`。

## 输出位置

手动页保留原来的 Output，且每次运行在其下创建独立目录。
智能页默认在输入 HDF5 所在目录下新建 `{文件名}_analysis_{时间}` 文件夹；可在 Settings 更改父目录。
完整成功的智能任务有 `analysis_summary.json`。失败或安全停止不会产生成功标记。
输入 HDF5 和 checkpoint 只读使用。

## 测试范围

发布记录中的自动验证只重建明确指定的测试切片，不代表已运行整个 2048 层体数据。
中心搜索仍覆盖原始全局范围并细化到 0.25 px。手动重建保留 Laminography，智能页当前只做 Tomography 的旋转中心查找。
