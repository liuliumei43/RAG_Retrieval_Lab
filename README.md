# RAG Retrieval Lab

一个适合阅读和复现的 Python 文档问答项目，重点学习检索设计、来源追踪和基础验证。

本项目基于 [Santanu Mohanta 的 chat-with-pdf-rag](https://github.com/santanu2908/chat-with-pdf-rag) 修改，采用 MIT 许可证，保留原作者版权与 Git 历史。原始实现、示例 PDF 和 19 项真实模型问答用例均来自上游；具体修改见 [修改说明](docs/修改说明.md)。

## 一句话解释

先把 PDF 切成带页码的小段，再结合语义和关键词找到相关段落，交给大模型组织回答，并返回检索来源。

```text
PDF → 提取文字和页码 → 分块 → 生成向量
问题 → 向量检索 + BM25 关键词检索 → RRF 融合 → 重排
     → 相关片段 + 用户问题 → 大模型 → 回答和来源
```

## 重点与难度

- 主线：PDF 解析、分块、向量检索、关键词检索、答案生成。
- 进阶：RRF 合并两个排名，cross-encoder 对候选段落重排。
- 工程部分：FastAPI 接口、流式输出、内存会话，以及无需模型密钥的回归测试。

核心代码集中在 `app/` 的七个 Python 文件。先读分块和检索，再读 API；无需从训练大模型开始。默认模型偏英文，样例是虚构的机器人产品资料；该项目不包含医疗数据或临床验证。

## 本版的小改进

1. PDF 改为内存读取，避免多个上传共用同一个临时文件。
2. BM25 仅让正分匹配参与排名融合，避免没有关键词命中的片段因零分排序获得额外票数。
3. 新增无需下载模型、无需 API Key 的回归测试；上游真实模型测试改为显式运行。
4. 增加中文阅读指南，修正重排说明与代码不一致、旧 Python 的流式字符串语法，以及完整会话的调试打印。

## 完整运行

建议 Python 3.12。安装依赖并配置自己的模型 Key：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install uv
uv sync
cp .env.example .env
# 编辑 .env，只配置自己选择的服务商
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs`，先调用 `/upload` 上传 PDF，再调用 `/query` 提问。首次检索会下载 embedding 和重排模型；答案生成需要所选服务商的有效 Key，费用与配额以该服务商为准。

## 仅验证核心逻辑

在干净的虚拟环境中安装测试依赖，运行：

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

这些测试使用真实 PDF 解析器、FAISS、BM25 与 FastAPI。向量由测试构造，LLM 响应用替身提供，验证程序行为，不衡量真实模型的理解能力或 RAG 回答准确率。

需要完整依赖、已下载模型和真实 API 的上游 19 项问答测试：

```bash
uv run pytest -o addopts= -m live tests/test_eval.py -v
```

上游 README 中的成功结果属于原作者报告，本仓库不将其视为本次实测结论。

## 阅读顺序

1. `app/pdf_loader.py`：文字和页码从哪里来。
2. `app/rag.py`：如何分块，如何获得语义向量。
3. `app/store.py`：FAISS、BM25 和 RRF 怎样配合。
4. `app/reranker.py`：为何只重排候选，保留哪些结果。
5. `app/main.py`：如何组装问题、上下文、对话与来源。
6. `app/llm.py`：模型服务接口。
7. `tests/test_core.py`：改动如何验证。

详细说明见 [中文学习指南](docs/学习指南.md)。

## 能力边界

一次只处理一份 PDF，索引和会话保存在内存，重启后丢失；没有 OCR、医疗领域微调或多用户隔离。返回的来源是检索候选，不能据此保证答案每一句都正确。RRF 分数不是置信概率；混合检索和重排是否有帮助，需要在固定问题集上实际比较。

## 来源与许可

- 上游：[santanu2908/chat-with-pdf-rag](https://github.com/santanu2908/chat-with-pdf-rag)
- 引入版本：`6fb4e9cfd0c190a950aadb49f7bb089fafe86e5e`
- 原始说明：[UPSTREAM_README](docs/UPSTREAM_README.md)
- 许可证：[MIT](LICENSE)，保留 Santanu Mohanta 的版权声明。
