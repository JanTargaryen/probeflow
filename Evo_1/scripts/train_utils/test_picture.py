from PIL import Image

img = Image.open('/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1/episode_1_20260228-115934/camera/rgb_1772251156516.jpg')
width, height = img.size
dpi = img.info.get('dpi')

print(f"Resolution: {width}x{height}")
print(f"DPI: {dpi}")