import os
import numpy as np
import kagglehub
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import class SVM tu Assignment 1
from svm import SoftMarginSVM

# ============================================================
# 1) LOAD & CHUẨN BỊ DỮ LIỆU (giong Assignment 1)
# ============================================================
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
data_dir  = os.path.join(path, "chest_xray")
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")
test_dir  = os.path.join(data_dir, "test")

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32

def make_dataset(directory, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )
    class_names = ds.class_names
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names

train_ds, class_names = make_dataset(train_dir, shuffle=True)
val_ds,   _           = make_dataset(val_dir)
test_ds,  _           = make_dataset(test_dir)
print("Class names:", class_names)

def ds_to_numpy(ds):
    X_list, y_list = [], []
    for imgs, labels in ds:
        X_list.append(imgs.numpy().reshape(imgs.shape[0], -1))
        y_list.append(labels.numpy().flatten())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

print("\nDang chuyen du lieu sang NumPy...")
X_train, y_train = ds_to_numpy(train_ds)
X_val,   y_val   = ds_to_numpy(val_ds)
X_test,  y_test  = ds_to_numpy(test_ds)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Nhan cho sklearn: {0, 1}
y_train_sk = y_train.astype(int)
y_val_sk   = y_val.astype(int)
y_test_sk  = y_test.astype(int)

# Nhan cho SVM numpy: {-1, +1}
y_train_svm = np.where(y_train == 0, -1, 1).astype(np.float64)
y_val_svm   = np.where(y_val   == 0, -1, 1).astype(np.float64)
y_test_svm  = np.where(y_test  == 0, -1, 1).astype(np.float64)


# ============================================================
# 2) CHUAN HOA FEATURE (dung cho ca 2 mo hinh)
# ============================================================
print("\nDang chuan hoa feature...")
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

X_train_sc = (X_train - mean) / std
X_val_sc   = (X_val   - mean) / std
X_test_sc  = (X_test  - mean) / std
print("Chuan hoa xong.")


# ============================================================
# 3) MODEL A — SKLEARN LinearSVC
# ============================================================
print("\n===== [MODEL A] SKLEARN LinearSVC =====")

# Tinh class_weight de xu ly mat can bang (giong Assignment 1)
sklearn_svm = LinearSVC(
    C=1.0,              # he so soft-margin
    max_iter=1000,      # so vong lap toi da
    class_weight="balanced",  # tu dong can bang class weight
    random_state=42,
)
sklearn_svm.fit(X_train_sc, y_train_sk)
print("Training xong.")


# ============================================================
# 4) MODEL B — SVM TU IMPLEMENT (Assignment 1)
# ============================================================
print("\n===== [MODEL B] SVM TU IMPLEMENT (Assignment 1) =====")

n_pos   = np.sum(y_train_svm ==  1)
n_neg   = np.sum(y_train_svm == -1)
n_total = len(y_train_svm)
w_pos   = n_total / (2 * n_pos)
w_neg   = n_total / (2 * n_neg)
sample_weights = np.where(y_train_svm == 1, w_pos, w_neg)

numpy_svm = SoftMarginSVM(
    C            = 1.0,
    lr           = 0.01,
    n_epochs     = 30,
    batch_size   = 64,
    lr_decay     = 0.95,
    clip_norm    = 5.0,
    random_state = 42,
)
numpy_svm.fit(
    X_train_sc, y_train_svm,
    sample_weights = sample_weights,
    X_val  = X_val_sc,
    y_val  = y_val_svm,
)


# ============================================================
# 5) DANH GIA & SO SANH
# ============================================================
def evaluate_sklearn(model, X, y_true, split_name):
    """Danh gia sklearn model, tra ve dict metrics."""
    y_pred = model.predict(X)
    return {
        "split"    : split_name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

def evaluate_numpy(model, X, y_true_svm, split_name):
    """Danh gia numpy SVM, doi nhan {-1,+1} -> {0,1} truoc khi tinh."""
    y_pred_svm = model.predict(X)
    y_pred = np.where(y_pred_svm == 1, 1, 0)
    y_true = np.where(y_true_svm == 1, 1, 0)
    return {
        "split"    : split_name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1"       : f1_score(y_true, y_pred, zero_division=0),
    }

# Thu thap ket qua
results_sklearn = [
    evaluate_sklearn(sklearn_svm, X_train_sc, y_train_sk, "Train"),
    evaluate_sklearn(sklearn_svm, X_val_sc,   y_val_sk,   "Val"),
    evaluate_sklearn(sklearn_svm, X_test_sc,  y_test_sk,  "Test"),
]
results_numpy = [
    evaluate_numpy(numpy_svm, X_train_sc, y_train_svm, "Train"),
    evaluate_numpy(numpy_svm, X_val_sc,   y_val_svm,   "Val"),
    evaluate_numpy(numpy_svm, X_test_sc,  y_test_svm,  "Test"),
]

# In bang so sanh
header = f"{'Split':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
sep    = "-" * 52

print("\n" + "=" * 52)
print("  [MODEL A] SKLEARN LinearSVC")
print("=" * 52)
print(header)
print(sep)
for r in results_sklearn:
    print(f"{r['split']:<8} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} "
          f"{r['Recall']:>10.4f} {r['F1']:>10.4f}")

print("\n" + "=" * 52)
print("  [MODEL B] SVM TU IMPLEMENT (NumPy SGD)")
print("=" * 52)
print(header)
print(sep)
for r in results_numpy:
    print(f"{r['split']:<8} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} "
          f"{r['Recall']:>10.4f} {r['F1']:>10.4f}")

# So sanh tren Test set
sk  = results_sklearn[-1]
num = results_numpy[-1]

print("\n" + "=" * 52)
print("  SO SANH TREN TEST SET")
print("=" * 52)
print(f"{'Metric':<12} {'Sklearn':>12} {'NumPy SGD':>12} {'Winner':>10}")
print("-" * 52)
for metric in ["Accuracy", "Precision", "Recall", "F1"]:
    sk_val  = sk[metric]
    num_val = num[metric]
    winner  = "Sklearn" if sk_val >= num_val else "NumPy SGD"
    print(f"{metric:<12} {sk_val:>12.4f} {num_val:>12.4f} {winner:>10}")

print("\nNhan xet:")
print("  - LinearSVC (sklearn): toi uu bang dual/primal LP, on dinh hon.")
print("  - NumPy SGD SVM: hoc tu gradient, phu thuoc nhieu vao lr va epoch.")
print("  - Ca hai deu dung soft-margin voi C=1.0 va class_weight balanced.")