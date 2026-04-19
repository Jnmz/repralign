# 最小比较示例

运行仓库内置 toy example：

```bash
python examples/minimal_compare.py
```

它演示了最基本的完整流程：

- 建两个 toy vision model
- hook candidate 和 reference 层
- 提取 pooled features
- 计算 layer-wise CKA
- 输出 CSV 和 PNG
