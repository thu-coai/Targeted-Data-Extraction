import os
import re

def clean_dir(dir):
    files = os.listdir(dir)

    print(dir)
    # print(files)

    file_losses = []
    for f in files:
        valloss = re.findall(r'valloss(\d+\.\d+)', f)[0]
        if len(valloss) == 0:
            print(dir, f)
        valloss = float(valloss)
        # print(float(valloss))
        file_losses.append([f, valloss])

    file_losses.sort(key=lambda x:x[1])
    if len(file_losses):
        print(dir, file_losses[0])
        for f, l in file_losses[1:]:
            path = os.path.join(dir, f)
            os.remove(path)

if __name__ == '__main__':
    for home, dirs, files in os.walk('./save'):
        # print(home)
        # print(dirs)
        if len(dirs) == 0:
            # home下只有文件
            clean_dir(home)
