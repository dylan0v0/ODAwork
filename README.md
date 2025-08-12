## 参数说明

- 'n': 样本数量。
- 'd': 特征维度（含截距）。
- 'b_set': 真实回归系数。
- 'sigma_set': 噪声标准差。
- 'gamma_set': 映射系数，用于生成映射空间变量。
- 'cu', 'co': 缺货成本和库存成本。
- 'z_g': 噪声分布，设为标准正态。


  ## 代码结构

- `get_data(n, d, b, sigma, gamma, z_g)`: 生成数据，包括设计矩阵 `v`，目标变量 `x`，以及映射后的变量 `x_reflect`。
- `ols(v, y)`: 使用普通最小二乘法拟合线性回归，返回回归系数和残差标准差。
- `q_boda_on_baseset(v0, v, b_hat_reflect, s_hat_reflect, cu, co)`: 计算关键分位点。
- `cost(q, demand, cu, co)`: 计算给定预测和真实需求的库存缺货成本。
- 主流程：生成数据、拟合模型、计算预测分位点，反向映射并计算成本。
