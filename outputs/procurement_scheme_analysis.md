# 采购方案结构性分析报告

本报告基于 `outputs/` 目录中 GA-greedy-only（优化后）与 manual（优化前）结果数据自动生成，重点回应审稿人关于供应商数量、跨基地协同、库存策略和优化前后协同效果展示的意见。

## 一、详细分析版本

### case_4x30

- 成本表现：manual 总成本为 50.36亿元，GA-greedy-only 总成本为 48.09亿元，相对变化为 -4.50%。当前数据中，GA-greedy-only 并未在该算例上获得成本优势，说明纯贪心解码 GA 的搜索结果仍有改进空间。
- 供应商结构：manual 启用 13 家供应商，GA-greedy-only 启用 14 家供应商；Top-5 采购份额分别为 69.47% 和 65.89%，HHI 分别为 0.1942 和 0.1810。
- 跨基地协同：manual 的多基地共享供应商采购占比为 100.00%，GA-greedy-only 为 61.67%；平均每个基地使用供应商数分别为 10.50 和 5.50。
- 库存策略：manual 平均安全库存冗余为 5.1887，GA-greedy-only 为 3.4766；库存成本占比分别为 5.66% 和 5.54%。
- 配矿与单位铁成本：manual 加权平均品位为 0.8060，GA-greedy-only 为 0.7928；单位铁元素综合成本分别为 1113.34 和 1136.97。

图形输出：
- `figures/case_4x30_cost_components.png`
- `figures/case_4x30_supplier_pareto.png`
- `figures/case_4x30_supplier_structure.png`
- `figures/case_4x30_cross_base_metrics.png`
- `figures/case_4x30_inventory_metrics.png`
- `figures/case_4x30_base_supplier_heatmap.png`
- `figures/case_4x30_inventory_lines.png`

### case_5x30

- 成本表现：manual 总成本为 58.66亿元，GA-greedy-only 总成本为 54.11亿元，相对变化为 -7.75%。当前数据中，GA-greedy-only 并未在该算例上获得成本优势，说明纯贪心解码 GA 的搜索结果仍有改进空间。
- 供应商结构：manual 启用 13 家供应商，GA-greedy-only 启用 15 家供应商；Top-5 采购份额分别为 73.50% 和 66.26%，HHI 分别为 0.1987 和 0.1877。
- 跨基地协同：manual 的多基地共享供应商采购占比为 100.00%，GA-greedy-only 为 87.57%；平均每个基地使用供应商数分别为 10.00 和 6.00。
- 库存策略：manual 平均安全库存冗余为 6.3574，GA-greedy-only 为 2.2446；库存成本占比分别为 6.25% 和 5.79%。
- 配矿与单位铁成本：manual 加权平均品位为 0.8212，GA-greedy-only 为 0.8007；单位铁元素综合成本分别为 1099.19 和 1118.10。

图形输出：
- `figures/case_5x30_cost_components.png`
- `figures/case_5x30_supplier_pareto.png`
- `figures/case_5x30_supplier_structure.png`
- `figures/case_5x30_cross_base_metrics.png`
- `figures/case_5x30_inventory_metrics.png`
- `figures/case_5x30_base_supplier_heatmap.png`
- `figures/case_5x30_inventory_lines.png`

## 二、用于论文对应章节的内容版本

为进一步揭示采购优化方案的结构性特征，本文从成本构成、供应商集中度、跨基地共享供应商、库存安全冗余和配矿质量等维度，对优化前后的采购方案进行了对比分析。结果表明，不同算法生成的采购方案不仅在总成本上存在差异，而且在供应商组合和基地间协同方式上呈现出明显不同的结构特征。
以 case_4x30 为例，优化前 manual 方案启用 13 家供应商，优化后 GA-greedy-only 方案启用 14 家供应商；其 Top-5 供应商采购份额由 69.47% 变为 65.89%，采购集中度 HHI 由 0.1942 变为 0.1810。跨基地协同方面，多基地共享供应商采购占比由 100.00% 变为 61.67%。库存策略方面，平均安全库存冗余由 5.1887 变为 3.4766。
以 case_5x30 为例，优化前 manual 方案启用 13 家供应商，优化后 GA-greedy-only 方案启用 15 家供应商；其 Top-5 供应商采购份额由 73.50% 变为 66.26%，采购集中度 HHI 由 0.1987 变为 0.1877。跨基地协同方面，多基地共享供应商采购占比由 100.00% 变为 87.57%。库存策略方面，平均安全库存冗余由 6.3574 变为 2.2446。
上述结果说明，采购方案优化会改变供应商选择、基地间供应商共享关系以及库存配置方式。本文据此增加供应商 Pareto 曲线、基地-供应商采购热力图、库存水平折线图和成本分项柱状图，以直观展示优化前后采购协同模式的差异。

## 三、审稿人回复版本

感谢审稿人的建议。根据该意见，本文补充了采购方案结构性分析，从供应商数量、供应商集中度、跨基地共享供应商、库存安全冗余和成本分项等角度对优化前后的采购方案进行比较。
具体而言，新增了供应商采购量 Pareto 图、基地-供应商采购热力图、库存水平折线图和成本分项柱状图，用于直观展示优化方案与人工经验方案在协同采购结构上的差异。
同时，本文在结果分析中报告了启用供应商数量、Top-5 供应商采购份额、Herfindahl 指数、多基地共享供应商采购占比、平均安全库存冗余和单位铁元素成本等指标，从而更清晰地说明采购优化方案在供应商组合、跨基地协同和库存策略方面的具体特征。

## 四、图表清单

### case_4x30
![figures/case_4x30_cost_components.png](figures/case_4x30_cost_components.png)
![figures/case_4x30_supplier_pareto.png](figures/case_4x30_supplier_pareto.png)
![figures/case_4x30_supplier_structure.png](figures/case_4x30_supplier_structure.png)
![figures/case_4x30_cross_base_metrics.png](figures/case_4x30_cross_base_metrics.png)
![figures/case_4x30_inventory_metrics.png](figures/case_4x30_inventory_metrics.png)
![figures/case_4x30_base_supplier_heatmap.png](figures/case_4x30_base_supplier_heatmap.png)
![figures/case_4x30_inventory_lines.png](figures/case_4x30_inventory_lines.png)

### case_5x30
![figures/case_5x30_cost_components.png](figures/case_5x30_cost_components.png)
![figures/case_5x30_supplier_pareto.png](figures/case_5x30_supplier_pareto.png)
![figures/case_5x30_supplier_structure.png](figures/case_5x30_supplier_structure.png)
![figures/case_5x30_cross_base_metrics.png](figures/case_5x30_cross_base_metrics.png)
![figures/case_5x30_inventory_metrics.png](figures/case_5x30_inventory_metrics.png)
![figures/case_5x30_base_supplier_heatmap.png](figures/case_5x30_base_supplier_heatmap.png)
![figures/case_5x30_inventory_lines.png](figures/case_5x30_inventory_lines.png)
