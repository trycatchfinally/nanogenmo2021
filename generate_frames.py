import argparse
import os

import cv2

print(cv2.__version__)

# this is a rough approximation of ffmpeg -i $vid.mp4 -r 1 -f image2 $vid-%4d.png

def extractImages(video_file, pathOut, interval, start_at = 0):
  count = start_at
  vidcap = cv2.VideoCapture(video_file)
  just_name = os.path.splitext(os.path.basename(video_file))[0]
  base_out = pathOut + "\\" + just_name
  if interval != 1000:
    base_out = base_out + '@' + str(interval)
  output_dir = base_out.replace('\\', '/')
  os.makedirs(base_out, exist_ok=True)
  print('base output directory: ', base_out)
  base_out = base_out + '\\' + just_name
  success, image = vidcap.read()
  if not success:
    assert False, video_file
  print(vidcap.get(cv2.CAP_PROP_FRAME_COUNT), 'frames total')
  print(vidcap.get(cv2.CAP_PROP_FPS), 'fps')
  while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * interval))  # added this line
    success, image = vidcap.read()
    if not success:
      break
    print(count, end=' ', flush=True)
    # print('Read frame #', count, ' at ', vidcap.get(cv2.CAP_PROP_POS_MSEC))
    cv2.imwrite(base_out + "-%04d.png" % count, image)  # save frame
    count = count + 1
  print('\ncompleted')
  return output_dir


if __name__ == "__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--video_file", help="path to video", default="example/video.mp4")
  a.add_argument("--output_path", help="path for image directory (example -> example/video/video-0000.png)", default='example')
  a.add_argument('--interval', help='interval in milliseconds', default=1000, type=int)
  args = a.parse_args()
  print(args)
  extractImages(args.video_file, args.output_path, args.interval)
