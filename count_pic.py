import os


def count_pic(path):
    files = os.listdir(path)
    count = len(files)
    return str(count)


if __name__ == "__main__":
    count_pic()
