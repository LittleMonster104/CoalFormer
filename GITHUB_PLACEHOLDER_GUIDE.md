# GitHub占位符和论文引用指南

## 📝 论文中的Code Availability部分

### 选项A：匿名审稿（推荐）

使用Anonymous GitHub进行匿名提交：

```latex
\section{Code and Data Availability}
\label{app:code}

\textbf{Code Repository:} For anonymous review, our implementation is available at:

\begin{center}
\url{https://anonymous.4open.science/r/coalformer-XXXX/}
\end{center}

The repository includes:
\begin{itemize}[leftmargin=*, itemsep=2pt]
    \item Complete PyTorch implementation of CoalFormer with learnable prototype clustering
    \item Training scripts for SMAC 3m with multiple random seeds
    \item Baseline implementations (VDN, QMIX, BRIDGE)
    \item Ablation study scripts (Table~\ref{tab:ablation})
    \item Visualization tools for coalition structure (Figure~\ref{fig:coalition_structure})
    \item Detailed documentation and hyperparameter configurations
\end{itemize}

\textbf{Pre-trained Models \& Data:} Training logs, model checkpoints, and experimental data will be publicly released upon acceptance.

\textbf{Reproducibility:} All experiments use fixed random seeds (42, 43, 44, 45). Expected training time: $\sim$45 minutes per seed on NVIDIA RTX 3090.
```

---

### 选项B：私有仓库（保守）

如果选择不在审稿期间公开代码：

```latex
\section{Code and Data Availability}
\label{app:code}

To ensure reproducibility, we will release:

\begin{itemize}[leftmargin=*, itemsep=2pt]
    \item Complete PyTorch implementation of CoalFormer
    \item Pre-trained model checkpoints for all experiments
    \item Training logs and evaluation metrics (TensorBoard format)
    \item Scripts to reproduce all figures and tables
    \item Detailed README with environment setup instructions
\end{itemize}

Code, pre-trained models, and data will be made publicly available upon acceptance at:
\begin{center}
\url{https://github.com/[ANONYMIZED]/coalformer}
\end{center}
```

---

### 选项C：公开仓库（接受后）

论文接受后使用的版本：

```latex
\section{Code and Data Availability}
\label{app:code}

\textbf{Code Repository:} Our complete implementation is available at:

\begin{center}
\url{https://github.com/YOUR_USERNAME/coalformer}
\end{center}

The repository includes:
\begin{itemize}[leftmargin=*, itemsep=2pt]
    \item Complete PyTorch implementation (\texttt{models/coalformer.py})
    \item Training scripts (\texttt{training/coalformer\_smac.py})
    \item Baseline methods (\texttt{baselines/})
    \item Ablation studies (\texttt{experiments/run\_ablation.py})
    \item Visualization tools (\texttt{analysis/})
    \item Comprehensive documentation (\texttt{README.md})
\end{itemize}

\textbf{Pre-trained Models:} Available at \url{https://huggingface.co/YOUR_USERNAME/coalformer-smac3m}

\textbf{Citation:}
\begin{verbatim}
@inproceedings{coalformer2027,
  title={CoalFormer: Learning Emergent Coalition Structures 
         via Differentiable Prototype Clustering},
  author={[Your Name]},
  booktitle={ICLR},
  year={2027}
}
\end{verbatim}
```

---

## 🔗 GitHub占位符列表

### 投稿时需要替换的占位符：

1. **[ANONYMIZED]** 
   - 用途：匿名审稿时隐藏作者身份
   - 替换为：实际的GitHub用户名（接受后）

2. **YOUR_USERNAME**
   - 用途：你的GitHub用户名
   - 替换为：你的实际GitHub账号名

3. **coalformer**
   - 用途：仓库名称
   - 建议保持不变

4. **XXXX**（在Anonymous GitHub链接中）
   - 用途：Anonymous GitHub自动生成的随机字符
   - 替换为：上传后获得的实际链接

---

## 📋 使用Anonymous GitHub的步骤

### 1. 访问Anonymous GitHub
```
https://anonymous.4open.science/
```

### 2. 上传代码
- 点击 "Submit Repository"
- 上传你的代码包或提供GitHub链接
- 获得匿名链接（格式：https://anonymous.4open.science/r/coalformer-XXXX/）

### 3. 在论文中使用
将获得的完整链接复制到论文的Code Availability部分

### 4. 注意事项
- ⚠️ Anonymous GitHub有时效限制（通常6个月）
- ⚠️ 确保上传前删除所有个人信息
- ⚠️ 可以随时更新代码

---

## 🎯 推荐的投稿策略

### ICLR 2027投稿

**阶段1：投稿时（现在）**
```latex
% 使用选项A
\url{https://anonymous.4open.science/r/coalformer-XXXX/}
```

**阶段2：审稿期间**
- 根据reviewer反馈更新代码
- 通过Anonymous GitHub链接保持匿名

**阶段3：接受后**
```latex
% 使用选项C
\url{https://github.com/YOUR_USERNAME/coalformer}
```
- 将私有仓库改为公开
- 更新论文中的链接
- 发布pre-trained models

---

## 📂 需要在论文LaTeX中替换的位置

### 主要位置：

1. **Appendix A (Code Availability)**
   ```latex
   \section{Code and Data Availability}
   \label{app:code}
   % 在这里使用上面的模板
   ```

2. **Abstract或Introduction（可选）**
   ```latex
   Code is available at \url{https://anonymous.4open.science/r/coalformer-XXXX/}
   ```

3. **Footnote（可选）**
   ```latex
   \footnote{Code: \url{https://anonymous.4open.science/r/coalformer-XXXX/}}
   ```

---

## ✅ 检查清单

投稿前确认：

- [ ] 选择了合适的Code Availability版本（A/B/C）
- [ ] 如果使用Anonymous GitHub，已上传代码并获得链接
- [ ] 替换了所有占位符（[ANONYMIZED]、YOUR_USERNAME、XXXX）
- [ ] 在论文中添加了正确的链接
- [ ] 代码仓库已删除所有个人信息
- [ ] README.md中包含安装和使用说明
- [ ] 测试过链接可以正常访问

---

## 💡 快速参考

**当前状态**：
- 代码包：`CoalFormer_Code_Release_v2/`
- 大小：35KB
- 文件：12个（9个Python + 3个文档）

**下一步**：
1. 决定使用选项A（匿名）还是选项B（私有）
2. 如果选A，上传到Anonymous GitHub
3. 将链接添加到论文Appendix
4. 提交论文

**推荐**：选项A（Anonymous GitHub），兼顾可复现性和匿名性
