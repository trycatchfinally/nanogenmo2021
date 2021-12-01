import os.path

import pytest

import cloud_vision
import generate_frames


def test_example():
  base_path = generate_frames.extractImages('example/Survival1951.mp4', 'example', 15000, start_at=2)
  assert base_path == 'example/Survival1951@15000'

  # this should take ~2 minutes the first time it's run
  # window = 4 is 'you can use this term 4 lines from this one' (so 1 = very next line, 0 = same as 1)
  # randomize order applies an unstable transform ('x', 'y', 'z' -> 'X, y. Z.' etc.)
  lines = cloud_vision.gen_for_directory(base_path, 'Survival 1951', filename='Survival-1951.md',
                                         window=4, randomize_order=True)
  cloud_vision.writelines('example/example-1951.md', lines)
  cloud_vision.run_pandoc('example/example-1951')
  assert os.path.isfile('example/example-1951.pdf')


def test_example_different_windows():
  # window 0 is same as window 1
  # need randomize order = False to make the files diff-able
  windows = [1, 2, 4, 8, 16, 32]
  base_path = generate_frames.extractImages('example/Survival1951.mp4', 'example', 15000, start_at=2)
  assert base_path == 'example/Survival1951@15000'
  word_count_for_window = {}
  for w in windows:
    lines = cloud_vision.gen_for_directory(base_path, 'Survival 1951', filename='1951@' + str(w) + '.md',
                                           window=w, randomize_order=False)
    cloud_vision.writelines('example/manual-eval-1951@' + str(w) + '.md', lines)
    word_count_for_window[w] = len(' '.join(lines).split(' '))

  assert word_count_for_window == {1: 591, 2: 519, 4: 467, 8: 421, 16: 393, 32: 368}


if __name__ == '__main__':
  pytest.main()
