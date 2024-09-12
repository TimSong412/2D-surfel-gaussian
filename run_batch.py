from pathlib import Path
import os

def run_dirs(base_dir):
    already_done = []
    with open('done.txt', 'r') as f:
        already_done = f.read().splitlines()
    
    all_dirs= sorted(Path(base_dir).iterdir())
    print(all_dirs)
    for dir in all_dirs:
        if dir.is_dir():
            if dir.__str__() in already_done:
                print(f"Skipping {dir}")
                continue
            cmd = f"python train.py -s {dir} --model_path {dir} --images color/light000 --iterations 1500"
            print(cmd)
            os.system(cmd)
            with open('done.txt', 'a') as f:
                f.write(f"{dir}\n")
            

if __name__ == '__main__':
    run_dirs('dataset/objv100')
    
