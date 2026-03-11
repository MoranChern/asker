# 项目组件

- `server.py`：Web 端主程序。
- `local_client.py`：本地命令行对话主程序。
- `index_manager.py`：索引管理程序。
- `graphrag.py`：图检索与图数据库处理模块。
- `agent.py`：对话代理模块。
- `gpu_worker.py`：GPU 工作模块。
- `qwen.py`：模型接入模块。
- `constants.py`：常量与配置模块。
- `static/`：静态资源目录。

# 需要删除数据库时

- 首先，确认docker已经down掉
- 然后使用下面的命令删除

```bash
docker run --rm -u 0:0 -v "$HOME/asker:/work" --entrypoint bash neo4j:4.4.11-community -lc 'rm -rf /work/neo_data'
```
