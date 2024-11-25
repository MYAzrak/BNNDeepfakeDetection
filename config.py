PROJECT_PATHS = {
    "test_dir": r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\test",
    "train_dir": r"C:\Users\monsi\Downloads\temp\CIFAKE-Real-and-AI-Generated-Synthetic-Images\DATASET\train",
}


def get_path(key):
    return PROJECT_PATHS.get(key, None)
