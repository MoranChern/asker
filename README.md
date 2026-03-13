# GraphRAG WebApp (no-index)

## 1) 文件结构

- `graphrag.py`：Neo4j 检索逻辑（无索引，直接按关键词扫描节点文本）
- `agent.py`：问答编排（先思考，再决定是否检索）
- `server.py`：Web 后端（FastAPI + WebSocket）
- `gpu_worker.py`：短生命周期 GPU 进程（llama.cpp/Qwen 推理 + JSONL 流式输出）
- `local_client.py`：命令行客户端（仅通过 Web 服务访问后端）
- `static/`：前端页面（滚动流 + 三种块 + Cytoscape 图谱证据）

## 3) Web 运行

```bash
python server.py
# 浏览器打开 http://localhost:8000
```

## 2) CLI 运行

```bash
# 需要线运行Web服务
python local_client.py
```

## 4) 说明

- 不再提供构建索引、维护索引、依赖索引的检索逻辑。
- 检索直接在 Neo4j 图中按 `Name/name/Description/description` 做关键词扫描。
- server 进程本身不 import `llama_cpp`，不会触碰 GPU。
- 每次回答会启动一次 `gpu_worker.py`，结束即退出释放显存。
- 证据以小图谱展示；点击节点/边会调用 `/api/node/{eid}` 或 `/api/rel/{rid}` 查看字段。
- Ctrl+Enter 发送；发送后输入块变提问块，下方生成回答块；回答结束后自动追加新的输入块。
