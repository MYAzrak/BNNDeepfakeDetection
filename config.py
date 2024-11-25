PROJECT_PATHS = {
    "test_dir": r"D:\MYA\AUS\Bachelor\Senior 2\COE 49413\Project\OursRepo\Dataset\test",
    "train_dir": r"D:\MYA\AUS\Bachelor\Senior 2\COE 49413\Project\OursRepo\Dataset\train",
}


def get_path(key):
    return PROJECT_PATHS.get(key, None)
