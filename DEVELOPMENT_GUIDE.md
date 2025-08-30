# CFR2005 项目开发指南

## 项目结构说明

这个项目使用 **git submodule** 的方式来管理 CFR (Climate Field Reconstruction) 核心代码：

```
cfr2005/                    # 主项目仓库
├── cfr/                    # CFR 核心代码 (git submodule)
│   ├── cfr/               # CFR Python 包
│   ├── docs/              # CFR 文档
│   ├── setup.py           # CFR 安装脚本
│   └── ...
├── CFR_LMR_Analysis.md     # 您的分析文档
├── cfr_test.py             # 测试脚本
├── lmr_workflow_example.py # 工作流示例
└── README.md               # 项目说明
```

## 开发环境设置

### 1. 克隆项目（包含 submodule）

如果其他人要克隆这个项目：

```bash
# 方法1：直接克隆包含 submodules
git clone --recursive https://github.com/simpupil/cfr2005.git

# 方法2：分步克隆
git clone https://github.com/simpupil/cfr2005.git
cd cfr2005
git submodule update --init --recursive
```

### 2. 设置 Python 环境

```bash
# 创建 conda 环境（推荐）
conda create -n cfr2005-dev python=3.10
conda activate cfr2005-dev

# 或者使用现有环境
conda activate cfr-env  # 您之前创建的环境
```

### 3. 安装 CFR 包（开发模式）

```bash
# 进入 cfr submodule 目录并安装
cd cfr
pip install -e .  # 开发模式安装
cd ..
```

## 开发工作流

### 使用 CFR 进行开发

现在您可以在项目中直接使用 CFR：

```python
# 在您的脚本中
import cfr
from cfr import ReconJob
from cfr.da import EnKF

# 创建重建作业
job = ReconJob()

# 使用现有的示例
# 参考 lmr_workflow_example.py 和 cfr_test.py
```

### 更新 CFR Submodule

当 CFR 原始仓库有更新时：

```bash
# 进入 cfr submodule
cd cfr

# 获取最新更新
git fetch origin
git merge origin/main  # 或 origin/master

# 返回主项目
cd ..

# 提交 submodule 更新
git add cfr
git commit -m "Update CFR submodule to latest version"
git push
```

### 开发您自己的代码

1. **在主项目根目录创建您的脚本**：
   ```python
   # my_analysis.py
   import sys
   import cfr
   from cfr import ReconJob
   
   def main():
       # 您的分析代码
       job = ReconJob()
       # ...
   
   if __name__ == "__main__":
       main()
   ```

2. **扩展现有示例**：
   - 修改 `lmr_workflow_example.py`
   - 创建新的分析脚本
   - 添加新的数据处理功能

## Git 工作流

### 提交您的更改

```bash
# 只提交主项目的更改（不包括 cfr submodule 内部更改）
git add .  # 这不会包含 cfr/ 内部的修改
git commit -m "Add new analysis features"
git push
```

### 如果需要修改 CFR 核心代码

**注意：通常不建议直接修改 submodule 代码，推荐的做法是：**

1. **Fork CFR 原始仓库** 到您自己的 GitHub 账号
2. **更新 submodule URL** 指向您的 fork：
   ```bash
   # 编辑 .gitmodules 文件
   vim .gitmodules
   # 将 url 改为：url = https://github.com/YOUR_USERNAME/cfr.git
   
   # 更新 submodule
   git submodule sync
   git submodule update --remote
   ```

3. **在您的 fork 中进行修改**，然后通过 Pull Request 贡献给原项目

## 最佳实践

### 1. 环境管理
- 使用 conda 或 virtualenv 管理 Python 环境
- 将依赖记录在 `requirements.txt` 或 `environment.yml` 中

### 2. 代码组织
- 将您的分析代码放在主项目根目录
- 使用有意义的文件名和文档
- 保持 CFR submodule 不变

### 3. 版本控制
- 定期更新 CFR submodule 以获取最新功能
- 为重要的开发里程碑创建 git tag
- 保持提交信息清晰明确

## 常见问题

### Q: submodule 文件夹是空的？
```bash
git submodule update --init --recursive
```

### Q: 如何查看 CFR 版本？
```bash
cd cfr
git log --oneline -5  # 查看最近5次提交
```

### Q: 如何重置 submodule？
```bash
git submodule deinit cfr
git submodule update --init cfr
```

## 示例项目

基于您现有的文件，您可以：

1. **运行测试**：`python cfr_test.py`
2. **运行工作流示例**：`python lmr_workflow_example.py`
3. **查看分析文档**：阅读 `CFR_LMR_Analysis.md`

这种 submodule 设置让您既能使用 CFR 的完整功能，又能保持自己项目的独立性和版本控制。
