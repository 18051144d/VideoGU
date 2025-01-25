import os
import sys
import glob
import json
import argparse
from PIL import Image, ImageDraw

from cnocr import CnOcr
from ultralytics import YOLO

class VALIDATOR:
    def __init__(self):
        self.ocr = CnOcr
        self.PoseDet = YOLO("yolo11m-pose.pt")

    def __call__(self, frame_path):
        # get human/non-human result
        full_pose_res = self.PoseDet(frame_path, save = False, verbose = False, conf = 0.5)[0]
        if full_pose_res.boxes.cls.nelement() == 0:
            return [], None, full_pose_res
        pose_res = [res.keypoints.xy.cpu().tolist() for res in full_pose_res]
        
        # mask chinese text
        OCR_res = self.ocr.ocr(frame_path)
        for res in OCR_res:
            res['position'] = res['position'].tolist()
        return pose_res, OCR_res, full_pose_res

def video2frame(videos_dir, dataset = 'nurvid', limit = -1):
    videos = glob.glob(os.path.join(videos_dir, '*.mp4'))
    for vi, video in enumerate(videos):
        frame_dir = f'./data/{dataset}/{os.path.basename(video)[:-4]}/frames'
        os.makedirs(frame_dir, exist_ok=True)
        os.system(f'ffmpeg -i {video} -vf fps=1 {frame_dir}/frame%04d.jpg')
        if vi == limit:
            break

def collect_all_valid(dataset = 'nurvid'):
    def collect_valid(frame_dir, processor):
        valid_frames, frames_info = {}, {}
        for frame_path in sorted(glob.glob(f'{frame_dir}/*.jpg')):
            pres, ores, _ = processor(frame_path)
            frames_info[os.path.basename(frame_path)] = {'pres': pres, 'ores': ores}
            if len(pres) == 0 and ores == None:
                continue
            valid_frames[os.path.basename(frame_path)] = frame_path.replace('frames', 'masked_frames')

        return valid_frames, frames_info
    
    def mask_text(valid_frames, frames_info, save_dir):
        def mask(frame_path, ores):
            frame = Image.open(frame_path).convert('RGB')
            draw = ImageDraw.Draw(frame)
            for res in ores:
                draw.rectangle([tuple(res['position'][0]), tuple(res['position'][2])], fill = 0)
            return frame

        for frame_key, frame_path in valid_frames.items():
            frame_info = frames_info[frame_key]
            masked_frame = mask(frame_path, frame_info['ores'])
            masked_frame.save(os.path.join(save_dir, frame_key))

    validator = VALIDATOR()
    for frame_dir in glob.glob(f'./data/{dataset}/*'):
        save_masked_dir = f'{frame_dir}/masked_frames'
        os.makedirs(save_masked_dir, exist_ok=True)

        valid_frames, frames_info = collect_valid(frame_dir, validator)
        mask_text(valid_frames, frames_info, save_masked_dir)

        with open(f"{frame_dir}/process_info.json", "w") as outfile: 
            json.dump({
                'valid_frames': valid_frames,
                'frames_info' : frames_info,
            }, outfile, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess video data')
    parser.add_argument('--videos_dir', type = str)
    parser.add_argument('--dataset', type = str, default = 'nurvid')
    parser.add_argument('--limit', type = int, default = -1)
    args = parser.parse_args()

    video2frame(args.videos_dir, args.dataset, args.limit)
    collect_all_valid(args.dataset)

