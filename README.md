# GraphRAG WebApp (split + streaming UI)

## 1) 文件结构
- `graphrag.py`：GraphRAG 业务逻辑（检索、索引、Prompt、Retriever…）
- `test.py`：CLI 主程序逻辑（原 graphrag_app 的 main loop）
- `server.py`：Web 后端（FastAPI + WebSocket）
- `gpu_worker.py`：短生命周期 GPU 进程（llama.cpp/Qwen 推理 + JSONL 流式输出）
- `static/`：前端页面（滚动流 + 三种块 + Cytoscape 图谱证据）

## 2) CLI 运行
```bash
python test.py
# 或构建索引/向量（写入 Neo4j）
python test.py --build-index
```

## 3) Web 运行
```bash
python server.py
# 浏览器打开 http://localhost:8000
```

## 4) 说明
- server 进程本身不 import `llama_cpp`，不会触碰 GPU。
- 每次回答会启动一次 `gpu_worker.py`，结束即退出释放显存。
- 证据以小图谱展示；点击节点/边会调用 `/api/node/{eid}` 或 `/api/rel/{rid}` 查看字段。
- Ctrl+Enter 发送；发送后输入块变提问块，下方生成回答块；回答结束后自动追加新的输入块。
