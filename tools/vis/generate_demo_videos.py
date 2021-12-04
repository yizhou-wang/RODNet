import os

overwrite = False
framerate = "20"
result_root = "results"
model_name = "rodnet-hg1wi-win16-wobg-mnet8-dcnv3-20200716-135222"
model_res_root = os.path.join(result_root, model_name)
video_root = os.path.join(result_root, model_name, "demos")
if not os.path.exists(video_root):
    os.makedirs(video_root)

for seq in sorted(os.listdir(model_res_root)):
    if not os.path.isdir(os.path.join(model_res_root, seq)):
        continue
    if seq == 'demos':
        continue
    img_root = os.path.join(model_res_root, seq, 'rod_viz')
    video_path = os.path.join(video_root, seq + "_demo.mp4")
    if not overwrite and os.path.exists(video_path):
        print("video exists. skip...")
        continue
    cmd = "ffmpeg -r " + framerate + " -i " + img_root + "/%010d.jpg -c:v libx264 -vf fps=" + framerate + \
          " -pix_fmt yuv420p " + video_path
    print(cmd)
    os.system(cmd)
