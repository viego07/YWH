[README.md](https://github.com/user-attachments/files/28292786/README.md)
# iris_knn（C 实现的 kNN 示例）

## 简介

这是一个用纯 C 编写的简单 kNN 分类器示例，基于 Iris 数据集的 4 维特征实现。程序完成数据加载、随机划分训练/测试集、按训练集统计量归一化、基于欧氏距离的 kNN 预测，并打印预测结果与准确率。

## 仓库结构

- `iris_knn.c`：主程序与所有实现。
- `data.txt`：示例数据文件（CSV 格式），请自行准备或替换。

## 数据格式

每行一条样本，逗号分隔，格式为：

```
sepal_length,sepal_width,petal_length,petal_width,label
```

label 支持的格式：
- 数字 `1`, `2`, `3`；
- 或字符串 `Iris-setosa` / `Iris-versicolor` / `Iris-virginica`；
- 或缩写 `setosa` / `versicolor` / `virginica`。

程序会忽略无法解析或标签不合法的行。

## 构建

在 macOS 或类 Unix 环境下使用 `clang` 或 `gcc` 编译：

```bash
clang -g iris_knn.c -o iris_knn
# 或：
# gcc -g iris_knn.c -o iris_knn
```

## 运行

用法：

```bash
./iris_knn [data_file] [k] [test_ratio]
```

- `data_file`：数据文件路径，默认 `data.txt`。
- `k`：kNN 的 k 值，默认 `5`。
- `test_ratio`：测试集占比（0~1），默认 `0.3`。

示例：

```bash
./iris_knn data.txt 5 0.3
```

程序输出包括训练/测试集标签分布、每个测试样本的实际标签与预测标签，以及最终的准确率统计。

## 实现要点

- 使用欧氏距离（Euclidean distance）作为相似度度量。
- 先随机打乱数据，再按比例划分训练/测试集；训练集统计量用于对训练集与测试集进行相同的归一化处理。
- k 值会被限定在不超过训练样本数范围内；投票平局时选择距离和较小的类别。

## 注意事项

- 程序对输入格式有基本的容错（会去除行末换行并跳过格式错误的行），但仍需保证每行至少年有四个数值特征与一个标签字段。
- 如果数据特征某列方差为 0，代码会将该列标准差设为 1 以避免除以 0。
- 随机数种子在代码中固定以便复现结果（可在 `iris_knn.c` 中修改）。

## 想要改进/扩展

- 支持 k-fold 交叉验证以更稳定地评估性能；
- 支持不同距离度量（如曼哈顿距离）；
- 支持标准输入或更灵活的数据加载器；
- 添加单元测试与 CI。

## 许可

此仓库当前未附带明确许可证；请根据需要为项目添加合适的许可文件（例如 `LICENSE`）。

## 贡献与反馈

欢迎提交 issue 或 pull request。若需要我代为编译并运行示例，请告诉我。
