# run_train.py
from Milestone2_Image_Understanding_Pipeline import TrainConfig, train_eval

cfg = TrainConfig(
    model_name="tiny_vit_11m_224",     # or "swin_tiny_patch4_window7_224"
    data_root="data",
    labels_csv="data/labels.csv",
    classes_txt="data/classes.txt",
    attributes_yaml="data/attributes.yaml",
    out_dir="outputs",

    # Your knobs:
    epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=0.03,
    warmup_epochs=10,
    label_smoothing_cls=0.05,

    # Loss weights (attributes)
    w_color=1.0,
    w_mat=1.0,
    w_cond=0.5,
    w_size=1.0,

    # System
    num_workers=8,
    seed=42,
    device="cpu"  # or "cpu"
)

train_eval(cfg)