# VideoGU
Video Grounding &amp; Understanding w/ LLM

# 配置
```zsh
conda deactivate
conda remove -n vidgu --all -y
conda clean -a -f
cd ~/.cache/pip
sudo rm -rf *
cd ~
```

```zsh
conda config --add channels defaults
conda update -n base conda -y
conda create -n vidgu python=3.12 -y
conda activate vidgu

# prompt: 有哪些包需要安装? 找出来, 然后在 README.md# 配置 中列出pip安装
# 基础依赖
pip install torch torchvision torchaudio  # 根据你的CUDA版本选择合适的torch版本
# LLM相关
pip install pydantic
pip install langchain-core
pip install langchain-google-genai  # Gemini API
pip install transformers
pip install accelerate
# 计算机视觉相关
pip install ultralytics  # YOLO
pip install cnocr  # 中文OCR
```

# 运行顺序和功能

prompt: 按照运行顺序, 介绍整个项目

基于代码分析，我来介绍整个项目的运行顺序和功能：

1. **数据预处理阶段** (`preprocess.py`)
   - 视频处理：
     - 使用`video2frame`函数将视频转换为帧（每秒1帧）
     - 帧保存在`./data/{dataset}/{video_name}/frames/`目录下
   - 帧验证和处理：
     - `VALIDATOR`类使用YOLO进行人体姿态检测
     - 使用CnOcr进行中文文本检测
     - 生成带掩码的帧（遮盖文本区域）

2. **LLM视觉理解阶段** (`models/`)
   - Gemini Flash模型 (`gemini_flash.py`)：
     - 处理单帧图像
     - 使用两轮评分系统：
       - 第一轮：返回前3个可能的程序代码（1-177）
       - 第二轮：返回最可能的程序代码（1-3）
   - QwenVL本地模型 (`qwenvl_local.py`)：
     - 提供本地部署的视觉语言模型选项

3. **评估和工具** (`utils.py`)
   - 提供评估指标：
     - IoU (Intersection over Union) 计算
     - Temporal IoU 计算
   - 用于验证模型预测的准确性

4. **数据组织结构**：
```
data/
├── nurvid/                # 数据集
│   ├── annotations/       # 标注数据
│   │   ├── procedures.txt # 程序代码定义
│   │   └── actions.txt    # 动作定义
│   └── {video_name}/      # 每个视频的数据
│       ├── frames/        # 原始帧
│       └── masked_frames/ # 处理后的帧
```

5. **提示词管理** (`prompts/`)
   - `nurvid_prompts.py`：存储用于视频理解的提示模板

这是一个视频理解项目，主要目标是：
1. 从视频中提取关键帧
2. 使用计算机视觉技术检测人体姿态和文本
3. 利用多模态LLM（Gemini/QwenVL）理解视频内容
4. 对视频内容进行分类和理解（使用预定义的程序代码体系）
