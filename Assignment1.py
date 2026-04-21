import os
import numpy as np
import kagglehub
import tensorflow as tf
from svm import SoftMarginSVM

# ============================================================
# 1) LOAD & CHUẨN BỊ DỮ LIỆU
# ============================================================
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

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
    class_names = ds.class_names   # luu TRUOC khi map/prefetch
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names

train_ds, class_names = make_dataset(train_dir, shuffle=True)
val_ds,   _           = make_dataset(val_dir)
test_ds,  _           = make_dataset(test_dir)
print("Class names:", class_names)


# ============================================================
# 2) CHUYEN DATASET -> NUMPY
# ============================================================
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

# SVM dung nhan {-1, +1}
y_train_svm = np.where(y_train == 0, -1, 1).astype(np.float64)
y_val_svm   = np.where(y_val   == 0, -1, 1).astype(np.float64)
y_test_svm  = np.where(y_test  == 0, -1, 1).astype(np.float64)

# ============================================================
# 2b) CHUAN HOA FEATURE
# ============================================================

print("\nDang chuan hoa feature (mean=0, std=1)...")
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8   # +epsilon tranh chia 0

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std    # dung mean/std cua train
X_test  = (X_test  - mean) / std
print("Chuan hoa xong.")




# ============================================================
# 4) HUAN LUYEN MO HINH
# ============================================================

# Tinh class weight xu ly mat can bang
n_pos   = np.sum(y_train_svm ==  1)
n_neg   = np.sum(y_train_svm == -1)
n_total = len(y_train_svm)
w_pos   = n_total / (2 * n_pos)
w_neg   = n_total / (2 * n_neg)
sample_weights = np.where(y_train_svm == 1, w_pos, w_neg)

print(f"\nClass weight -> PNEUMONIA: {w_pos:.3f} | NORMAL: {w_neg:.3f}")
print(f"So mau       -> PNEUMONIA: {n_pos}      | NORMAL: {n_neg}")

print("\n===== BAT DAU HUAN LUYEN SVM =====")
svm = SoftMarginSVM(
    C            = 1.0,
    lr           = 0.01,   
    n_epochs     = 30,
    batch_size   = 64,
    lr_decay     = 0.95,
    clip_norm    = 5.0,    # gradient clipping
    random_state = 42,
)
svm.fit(
    X_train, y_train_svm,
    sample_weights = sample_weights,
    X_val  = X_val,
    y_val  = y_val_svm,
)

# Kiem tra sau train
print("\nKiem tra w:", "NaN!" if np.isnan(svm.w).any() else "OK")
print("Predict phan bo tren test set:")
preds = svm.predict(X_test)
print(f"  PNEUMONIA (+1): {np.sum(preds==1)}")
print(f"  NORMAL    (-1): {np.sum(preds==-1)}")


# ============================================================
# 5) DANH GIA MO HINH: Precision, Recall, F1
# ============================================================
def compute_metrics(y_true, y_pred, positive_label=1):
    TP = np.sum((y_pred == positive_label) & (y_true == positive_label))
    FP = np.sum((y_pred == positive_label) & (y_true != positive_label))
    FN = np.sum((y_pred != positive_label) & (y_true == positive_label))
    TN = np.sum((y_pred != positive_label) & (y_true != positive_label))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / len(y_true)
    return dict(Precision=precision, Recall=recall, F1=f1,
                Accuracy=accuracy, TP=TP, FP=FP, FN=FN, TN=TN)

def print_metrics(split_name, y_true, y_pred):
    m = compute_metrics(y_true, y_pred, positive_label=1)
    print(f"\n{'='*45}")
    print(f"  {split_name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {m['Accuracy']:.4f}")
    print(f"  Precision : {m['Precision']:.4f}")
    print(f"  Recall    : {m['Recall']:.4f}")
    print(f"  F1-Score  : {m['F1']:.4f}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    return m

print("\n===== KET QUA DANH GIA =====")
print_metrics("TRAIN SET",      y_train_svm, svm.predict(X_train))
print_metrics("VALIDATION SET", y_val_svm,   svm.predict(X_val))
print_metrics("TEST SET",       y_test_svm,  svm.predict(X_test))
print()
