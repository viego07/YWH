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

        token = strtok_s(line, ",", &context);
        for (int i = 0; i < FEATURE_DIM; i++) {
            if (token == NULL) {
                valid = 0;
                break;
            }
            values[i] = atof(token);
            token = strtok_s(NULL, ",", &context);
        }

        if (!valid || token == NULL) {
            continue;
        }

        char *label_text = token;
        label_text[strcspn(label_text, "\r\n")] = '\0';

        int label = parse_label(label_text);
        if (label == 0) {
            continue;
        }

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

static void shuffle_samples(Sample *samples, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Sample temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
    }
}

static void compute_mean_std(const Sample *samples, int n, double mean[FEATURE_DIM], double stddev[FEATURE_DIM]) {
    for (int i = 0; i < FEATURE_DIM; i++) {
        mean[i] = 0.0;
        stddev[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            mean[j] += samples[i].feature[j];
        }
    }

    for (int j = 0; j < FEATURE_DIM; j++) {
        mean[j] /= n;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            double diff = samples[i].feature[j] - mean[j];
            stddev[j] += diff * diff;
        }
    }

    for (int j = 0; j < FEATURE_DIM; j++) {
        stddev[j] = sqrt(stddev[j] / n);
        if (stddev[j] == 0.0) {
            stddev[j] = 1.0;
        }
    }
}

static void normalize_samples(Sample *samples, int n, const double mean[FEATURE_DIM], const double stddev[FEATURE_DIM]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            samples[i].feature[j] = (samples[i].feature[j] - mean[j]) / stddev[j];
        }
    }
}

static double euclidean_distance(const double a[FEATURE_DIM], const double b[FEATURE_DIM]) {
    double sum = 0.0;
    for (int i = 0; i < FEATURE_DIM; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

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

static int knn_predict(const Sample *train, int train_n, const Sample *sample, int k) {
    Neighbor *neighbors = (Neighbor *)malloc((size_t)train_n * sizeof(Neighbor));
    if (neighbors == NULL) {
        return 0;
    }

    for (int i = 0; i < train_n; i++) {
        neighbors[i].distance = euclidean_distance(sample->feature, train[i].feature);
        neighbors[i].label = train[i].label;
    }

    qsort(neighbors, (size_t)train_n, sizeof(Neighbor), compare_neighbor);

    int vote[4] = {0};
    double dist_sum[4] = {0.0};
    int limit = k < train_n ? k : train_n;

    for (int i = 0; i < limit; i++) {
        int label = neighbors[i].label;
        if (label >= 1 && label <= 3) {
            vote[label]++;
            dist_sum[label] += neighbors[i].distance;
        }
    }

    free(neighbors);

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