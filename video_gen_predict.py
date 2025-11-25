import os, subprocess, shlex
result_num = 1
HOME = os.getcwd()
EXPRESSION_PATH = os.path.join(HOME, f"refer-kitti/expression/") 
GT_PATH = os.path.join(HOME, f"refer-kitti/KITTI/labels_with_ids/image_02/")
EXP_PATH = os.path.join(HOME, f"exps/results_{result_num}/")
LABELS_PATH = os.path.join(HOME, f"refer-kitti/KITTI/labels_with_ids/image_02/")
KITTI_PATH = f"{HOME}/refer-kitti/KITTI/training/image_02/"
 
EXP_NUM="0013"
expression = "women"
images_dir = os.path.join(KITTI_PATH, EXP_NUM)
predict_file = os.path.join(EXP_PATH, EXP_NUM, expression, "gt.txt")

# Try to check if the gt file exists and print a few lines
try:
    if os.path.exists(predict_file):
        print(f"GT file exists: {predict_file}")
        with open(predict_file, 'r') as f:
            lines = f.readlines()[:5]
            print(f"First 5 lines of GT file:")
            for line in lines:
                print(line.strip())
    else:
        print(f"GT file does NOT exist: {predict_file}")
except Exception as e:
    print(f"Error while reading GT file: {e}")

print(f"\nImages directory: {images_dir}")
print(f"Output directory: tracking_viz/predict_viz")

cmd = f'python draw_boxes.py --gt_file "{predict_file}" --images_dir "{images_dir}" --output_dir tracking_viz/predict_viz'
print(f"\nRunning: {cmd}")
proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
print(proc.stdout)
if proc.returncode != 0:
    print("Return code:", proc.returncode)
    print(proc.stderr)
