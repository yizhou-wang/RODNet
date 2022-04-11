import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, help='directory of RODNet results')
    parser.add_argument('--framerate', default='20', type=str, help='video framerate')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing videos')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    overwrite = args.overwrite
    framerate = args.framerate
    result_dir = args.result_dir
    result_root = os.path.dirname(result_dir)
    model_name = os.path.basename(result_dir)
    video_root = os.path.join(result_root, model_name, "demos")
    if not os.path.exists(video_root):
        os.makedirs(video_root)

    for seq in sorted(os.listdir(result_root)):
        if not os.path.isdir(os.path.join(result_root, seq)):
            continue
        if seq == 'demos':
            continue
        img_root = os.path.join(result_root, seq, 'rod_viz')
        video_path = os.path.join(video_root, seq + "_demo.mp4")
        if not overwrite and os.path.exists(video_path):
            print("video exists. skip...")
            continue
        cmd = "ffmpeg -r " + framerate + " -i " + img_root + "/%010d.jpg -c:v libx264 -vf fps=" + framerate + \
              " -pix_fmt yuv420p " + video_path
        print(cmd)
        os.system(cmd)
