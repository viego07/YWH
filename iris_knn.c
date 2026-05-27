#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define FEATURE_DIM 4
#define MAX_LINE 512
#define DEFAULT_K 5
#define DEFAULT_TEST_RATIO 0.3

typedef struct {
    double feature[FEATURE_DIM];
    int label;
} Sample;

typedef struct {
    double distance;
    int label;
} Neighbor;

/*
 * 函数：parse_label
 * 摘要：将数据文本中的标签解析为整数编码（1=setosa, 2=versicolor, 3=virginica）。
 * 参数：
 *   - text: 指向标签文本的字符串指针（可能为"Iris-setosa"等或数字字符）。
 * 返回：解析得到的标签编号，未识别返回0。
 */
static int parse_label(const char *text) {
    if (text == NULL) {
        return 0;
    }

    while (isspace((unsigned char)*text)) {
        text++;
    }

    if (*text == '\0') {
        return 0;
    }

    if (strcmp(text, "1") == 0 || strcasecmp(text, "Iris-setosa") == 0 ||
        strcasecmp(text, "setosa") == 0) {
        return 1;
    }
    if (strcmp(text, "2") == 0 || strcasecmp(text, "Iris-versicolor") == 0 ||
        strcasecmp(text, "versicolor") == 0) {
        return 2;
    }
    if (strcmp(text, "3") == 0 || strcasecmp(text, "Iris-virginica") == 0 ||
        strcasecmp(text, "virginica") == 0) {
        return 3;
    }

    return 0;
}

/*
 * 函数：load_dataset
 * 摘要：从CSV文件中读取数据集，每行包含FEATURE_DIM个特征和一个标签。
 * 参数：
 *   - filename: 数据文件路径
 *   - samples_out: 输出参数，返回分配的样本数组指针
 * 返回：样本数量（成功）或-1（失败）
 */
static int load_dataset(const char *filename, Sample **samples_out) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return -1;
    }

    int capacity = 128;
    int size = 0;
    Sample *samples = (Sample *)malloc((size_t)capacity * sizeof(Sample));
    if (samples == NULL) {
        fclose(fp);
        return -1;
    }

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *token;
        char *context = NULL;
        double values[FEATURE_DIM];
        int valid = 1;
        /* 使用 strtok_r 按逗号分割行，读取 FEATURE_DIM 个特征值 */
        token = strtok_r(line, ",", &context);
        for (int i = 0; i < FEATURE_DIM; i++) {
            if (token == NULL) {
                valid = 0;
                break;
            }
            values[i] = atof(token);
            token = strtok_r(NULL, ",", &context);
        }

        if (!valid || token == NULL) {
            continue;
        }

        /* 去除标签字段末尾的换行符并解析标签文本 */
        char *label_text = token;
        label_text[strcspn(label_text, "\r\n")] = '\0';

        int label = parse_label(label_text); /* 解析标签为整数编码 */
        if (label == 0) {
            continue;
        }

        /* 动态扩容数组（若需要） */
        if (size >= capacity) {
            capacity *= 2;
            Sample *resized = (Sample *)realloc(samples, (size_t)capacity * sizeof(Sample));
            if (resized == NULL) {
                free(samples);
                fclose(fp);
                return -1;
            }
            samples = resized;
        }

        /* 将解析到的特征与标签写入样本数组 */
        for (int i = 0; i < FEATURE_DIM; i++) {
            samples[size].feature[i] = values[i];
        }
        samples[size].label = label;
        size++;
    }

    fclose(fp);
    *samples_out = samples;
    return size;
}

/*
 * 函数：shuffle_samples
 * 摘要：使用 Fisher–Yates 算法随机打乱样本顺序，原地修改数组。
 * 参数：
 *   - samples: 要打乱的样本数组
 *   - n: 数组长度
 */
static void shuffle_samples(Sample *samples, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1); /* 随机选择索引 j */
        Sample temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp; /* 交换 i 和 j */
    }
}

/*
 * 函数：compute_mean_std
 * 摘要：计算每个特征的均值与标准差（无偏估计使用样本均方根），用于后续归一化。
 * 参数：
 *   - samples: 样本数组（只读）
 *   - n: 样本数量
 *   - mean: 输出数组，长度为 FEATURE_DIM，返回每个特征的均值
 *   - stddev: 输出数组，长度为 FEATURE_DIM，返回每个特征的标准差（若为0则设为1以避免除0）
 */
static void compute_mean_std(const Sample *samples, int n, double mean[FEATURE_DIM], double stddev[FEATURE_DIM]) {
    for (int i = 0; i < FEATURE_DIM; i++) {
        mean[i] = 0.0;
        stddev[i] = 0.0;
    }

    /* 累加求和以计算均值 */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            mean[j] += samples[i].feature[j];
        }
    }

    for (int j = 0; j < FEATURE_DIM; j++) {
        mean[j] /= n; /* 均值 = 总和 / 样本数 */
    }

    /* 计算方差的累加项：sum((x - mean)^2) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            double diff = samples[i].feature[j] - mean[j];
            stddev[j] += diff * diff;
        }
    }

    /* 开方并处理零方差情况 */
    for (int j = 0; j < FEATURE_DIM; j++) {
        stddev[j] = sqrt(stddev[j] / n);
        if (stddev[j] == 0.0) {
            stddev[j] = 1.0; /* 防止除以0，若特征恒定则不做缩放 */
        }
    }
}

/*
 * 函数：normalize_samples
 * 摘要：对样本数组按给定均值和标准差进行零均值单位方差归一化（z-score）。
 * 参数：
 *   - samples: 待归一化的样本数组（原地修改）
 *   - n: 样本数量
 *   - mean: 每个特征的均值
 *   - stddev: 每个特征的标准差
 */
static void normalize_samples(Sample *samples, int n, const double mean[FEATURE_DIM], const double stddev[FEATURE_DIM]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            /* 归一化公式：(x - mean) / stddev */
            samples[i].feature[j] = (samples[i].feature[j] - mean[j]) / stddev[j];
        }
    }
}

/*
 * 函数：euclidean_distance
 * 摘要：计算两个特征向量之间的欧氏距离。
 * 参数：
 *   - a: 第一个向量
 *   - b: 第二个向量
 * 返回：欧氏距离
 */
static double euclidean_distance(const double a[FEATURE_DIM], const double b[FEATURE_DIM]) {
    double sum = 0.0;
    for (int i = 0; i < FEATURE_DIM; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff; /* 累加差的平方 */
    }
    return sqrt(sum); /* 返回欧氏距离 = sqrt(sum((a-b)^2)) */
}

/*
 * 函数：compare_neighbor
 * 摘要：用于 qsort 的比较器，根据距离升序排序邻居。
 * 参数：两个指向 Neighbor 的指针
 * 返回：与 qsort 约定一致的比较结果
 */
static int compare_neighbor(const void *lhs, const void *rhs) {
    const Neighbor *a = (const Neighbor *)lhs;
    const Neighbor *b = (const Neighbor *)rhs;
    if (a->distance < b->distance) {
        return -1;
    }
    if (a->distance > b->distance) {
        return 1;
    }
    return 0;
}

/*
 * 函数：knn_predict
 * 摘要：对单个样本使用kNN进行预测。
 * 参数：
 *   - train: 训练样本数组
 *   - train_n: 训练样本数
 *   - sample: 待预测样本
 *   - k: 邻居数
 * 返回：预测标签（1/2/3）
 */
static int knn_predict(const Sample *train, int train_n, const Sample *sample, int k) {
    Neighbor *neighbors = (Neighbor *)malloc((size_t)train_n * sizeof(Neighbor));
    if (neighbors == NULL) {
        return 0; /* 内存分配失败返回0（未定义标签） */
    }

    /* 计算每个训练样本到待测样本的距离并记录标签 */
    for (int i = 0; i < train_n; i++) {
        neighbors[i].distance = euclidean_distance(sample->feature, train[i].feature);
        neighbors[i].label = train[i].label;
    }

    /* 按距离对邻居排序，最近的排在前面 */
    qsort(neighbors, (size_t)train_n, sizeof(Neighbor), compare_neighbor);

    /* 投票计数与距离和（用于平票时以较小距离获胜） */
    int vote[4] = {0};
    double dist_sum[4] = {0.0};
    int limit = k < train_n ? k : train_n; /* k 不得超过训练样本数 */

    for (int i = 0; i < limit; i++) {
        int label = neighbors[i].label;
        if (label >= 1 && label <= 3) {
            vote[label]++;
            dist_sum[label] += neighbors[i].distance; /* 累加用于平票比较 */
        }
    }

    free(neighbors);

    /* 选择票数最多的标签，若票数相等则选择距离和较小的标签 */
    int best_label = 1;
    for (int label = 2; label <= 3; label++) {
        if (vote[label] > vote[best_label]) {
            best_label = label;
        } else if (vote[label] == vote[best_label] && dist_sum[label] < dist_sum[best_label]) {
            best_label = label;
        }
    }

    return best_label;
}

/*
 * 函数：print_label_distribution
 * 摘要：打印样本集中各类别的数量分布，用于检查训练/测试集是否均衡。
 * 参数：
 *   - samples: 样本数组
 *   - n: 样本数
 *   - name: 集合名称（例如"Training set"）
 */
static void print_label_distribution(const Sample *samples, int n, const char *name) {
    int count[4] = {0};
    for (int i = 0; i < n; i++) {
        if (samples[i].label >= 1 && samples[i].label <= 3) {
            count[samples[i].label]++;
        }
    }

    printf("%s: total=%d, setosa=%d, versicolor=%d, virginica=%d\n",
           name, n, count[1], count[2], count[3]);
}

/*
 * 函数：main
 * 摘要：程序入口，负责参数解析、加载数据、划分训练/测试集、归一化、训练并评估kNN模型。
 * 可选参数：
 *   - argv[1]: 数据文件路径（默认 data.txt）
 *   - argv[2]: k 值（默认 5）
 *   - argv[3]: 测试集比例（0~1，默认 0.3）
 */
int main(int argc, char *argv[]) {
    const char *filename = (argc >= 2) ? argv[1] : "data.txt";
    int k = (argc >= 3) ? atoi(argv[2]) : DEFAULT_K;
    double test_ratio = (argc >= 4) ? atof(argv[3]) : DEFAULT_TEST_RATIO;

    if (k <= 0) {
        k = DEFAULT_K;
    }
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        test_ratio = DEFAULT_TEST_RATIO;
    }

    Sample *samples = NULL;
    int total = load_dataset(filename, &samples);
    if (total <= 0) {
        fprintf(stderr, "Failed to load dataset from %s\n", filename);
        return 1;
    }

    srand(2026);
    shuffle_samples(samples, total);

    int test_n = (int)(total * test_ratio);
    if (test_n < 1) {
        test_n = 1;
    }
    if (test_n >= total) {
        test_n = total / 3;
        if (test_n < 1) {
            test_n = 1;
        }
    }

    int train_n = total - test_n;
    Sample *train = samples;
    Sample *test = samples + train_n;

    double mean[FEATURE_DIM];
    double stddev[FEATURE_DIM];
    compute_mean_std(train, train_n, mean, stddev);
    normalize_samples(train, train_n, mean, stddev);
    normalize_samples(test, test_n, mean, stddev);

    print_label_distribution(train, train_n, "Training set");
    print_label_distribution(test, test_n, "Test set");

    int correct = 0;
    for (int i = 0; i < test_n; i++) {
        int predicted = knn_predict(train, train_n, &test[i], k);
        if (predicted == test[i].label) {
            correct++;
        }
        printf("Test %3d: actual=%d predicted=%d\n", i + 1, test[i].label, predicted);
    }

    double accuracy = (double)correct / test_n;
    printf("\nK = %d\n", k);
    printf("Train samples = %d\n", train_n);
    printf("Test samples  = %d\n", test_n);
    printf("Correct       = %d\n", correct);
    printf("Accuracy      = %.2f%%\n", accuracy * 100.0);

    free(samples);
    return 0;
}