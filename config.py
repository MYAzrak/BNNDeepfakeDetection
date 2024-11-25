PROJECT_PATHS = {
    "test_dir": r"D:\MYA\AUS\Bachelor\Senior 2\COE 49413\Project\OursRepo\Dataset\test",
    "train_dir": r"D:\MYA\AUS\Bachelor\Senior 2\COE 49413\Project\OursRepo\Dataset\train",
    "model_path": r"D:\MYA\AUS\Bachelor\Senior 2\COE 49413\Project\OursRepo\CV-Project-G5\best_model.pth",
}


def get_path(key):
    return PROJECT_PATHS.get(key, None)
