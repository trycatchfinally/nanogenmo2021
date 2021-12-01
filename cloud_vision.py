import atexit
import glob
import hashlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
from collections import Counter
from typing import Iterable, Optional, List, Tuple

import jsonpickle
import pandas as pd
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ImageAnalysis
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from loguru import logger
from msrest.authentication import CognitiveServicesCredentials
from tqdm.auto import tqdm

import time

from ratelimiter import RateLimiter

import secrets
import textmods

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = secrets.VISION_KEY
endpoint = "https://genmo2021.cognitiveservices.azure.com/"
# not all of these are used but it's the free tier and it makes caching easier to have everything there
all_visual_features = [eval('VisualFeatureTypes.' + x) for x in dir(VisualFeatureTypes) if "__" not in x]

# noinspection PyTypeChecker
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def persist_to_file(file_name):
  try:
    cache = json.load(open(file_name, 'r'))
  except (IOError, ValueError):
    cache = {}

  atexit.register(lambda: json.dump(cache, open(file_name, 'w'), indent=2))

  def decorator(func):
    def new_func(param):
      assert isinstance(param, str), ('only strings supported due to json round-tripping, got:', type(param), param)
      if param not in cache:
        cache[param] = func(param)
      return cache[param]

    return new_func

  return decorator


@persist_to_file('.cache/_get_file_hash.json')
def get_file_hash(filename: str) -> str:
  hash_alg = hashlib.sha256()
  block_size = 65536
  with open(filename, 'rb') as f:
    chunk = f.read(block_size)
    while len(chunk) > 0:
      hash_alg.update(chunk)
      chunk = f.read(block_size)
  return hash_alg.hexdigest()


def _limited(until):
  duration = round(until - time.time(), 2)
  if duration > 1.0:
    logger.info('Rate limited, sleeping {:2.2f} seconds', duration)


rate_limiter = RateLimiter(max_calls=20, period=61, callback=_limited)


def assert_phrases(phrases: List[str]) -> List[str]:
  for i, p in enumerate(phrases):
    assert p, ('empty', i, phrases)
    # if ' ' not in p: assert p.islower(), (p, phrases)
    if not p.endswith(' et al.'):
      assert not p.endswith('.'), (p, i, phrases)
    assert not p.endswith(','), (p, i, phrases)
  return phrases


def is_overlap(phrases: List[str], to_find: str) -> bool:
  sfs = ' ' + to_find + ' '
  sf = ' ' + to_find
  fs = to_find + ' '
  for p in phrases:
    if p == to_find:
      continue
    if to_find not in p:
      continue
    if p.startswith(fs):
      return True
    if p.endswith(sf):
      return True
    if sfs in p:
      return True
  return False


def is_overlap_or_exact(phrases: List[str], to_find: str) -> bool:
  if to_find in phrases:
    return True
  return is_overlap(phrases, to_find)


known_compounds = [
  {'bed', 'room'},
  {'bath', 'tub'},
  {'road', 'way'},
  {'business', 'person'},
  {'house', 'plant'}
]


def _reducer(phrases: List[str]) -> Tuple[List[str], List[str]]:
  assert_phrases(phrases)
  kept, removed = [], []
  more_generic_terms = (set(itertools.chain.from_iterable(get_generic_terms_for(w) for w in phrases)))

  compound_words_list = [(x, y) for x in phrases for y in phrases if x + y in phrases]
  compound_words = set(itertools.chain(*compound_words_list))
  if compound_words:
    assert compound_words in known_compounds, (compound_words, phrases)

  for i, k in enumerate(phrases):
    drop = False
    drop = drop or k in kept

    if not drop and k in compound_words:
      logger.trace('dropping compound: {} {} from {}', i, k, phrases)
      drop = True

    if not drop and k in more_generic_terms:
      logger.trace('dropping generic: {} {} from {}', i, k, phrases)
      drop = True

    drop = drop or is_overlap(phrases, k)

    for suff in ['s', 'ing', 'ness']:
      if not drop and is_overlap_or_exact(phrases, k + suff):
        logger.trace('drop {}: {} {} from {}', suff, i, k, phrases)
        drop = True
        if suff == 'ness':
          assert k == 'dark'

    if drop:
      removed.append(k)
      continue
    kept.append(k)
  return kept, removed


def reduce(phrases: List[str]) -> List[str]:
  # r_copy = list(reversed(phrases))
  kept, removed = _reducer(phrases)
  # r_kept, r_removed = reducer(r_copy)
  # r_kept.reverse()
  # assert kept == r_kept
  return kept


def fast_isfile(filename: str) -> bool:
  return os.path.isfile(filename)


# there are some weird json round-tripping issues with the Azure API's so it's safest to use python binary pickles
def analyze(filename: str) -> Optional[ImageAnalysis]:
  cache_file = '.cache/' + get_file_hash(filename) + '.pkl'
  response: ImageAnalysis
  if fast_isfile(cache_file):
    with open(cache_file, 'rb') as source:
      response = pickle.load(source)
  else:
    with rate_limiter:
      response = computervision_client.analyze_image_in_stream(open(filename, 'rb'), visual_features=all_visual_features)
    # save the python binary pickle for future use
    with open(cache_file, 'wb') as output:
      pickle.dump(response, output)

  # for debugging purposes, we save a jsonpickle with some lines removed
  # this is used in get_hierarchy
  debug_json = '.debug/' + os.path.splitext(os.path.basename(filename))[0] + '.json'
  if not fast_isfile(debug_json):
    with open(debug_json, 'wt') as dj:
      dj.writelines(filter_json(jsonpickle.encode(response, indent=4)))
  return response


def filter_json(block: str) -> Iterable[str]:
  for line in block.split('\n'):
    if 'py/object' in line:
      assert line.strip().endswith(','), line
      continue
    if ': {},' in line:      continue
    yield line + '\n'


all_celebs = set()


# converts the full response to a list of phrases (including celebrities, description, etc.)
def extract_text(d: ImageAnalysis, filename_for_debugging: str = '') -> List[str]:
  def _l(s: str):
    if s == "Petri dish":
      return s.lower()
    if s != "LEGO":
      assert s.islower(), s
    return s

  x = []

  def _add(s):
    nonlocal x
    if s not in x:
      x.append(s)

  found = False
  for c in d.description.captions:
    found = True
    _add(c.text)  # not using c.confidence

  assert found, d.description
  for cat in d.categories:
    if cat.detail and cat.detail.celebrities:
      for celeb in cat.detail.celebrities:
        all_celebs.add(celeb.name)
        _add(celeb.name)
  for t in d.description.tags:
    _add(_l(t))
  for tx in d.tags:
    _add(_l(tx.name))
  for o in d.objects:
    _add(_l(o.object_property.lower()))

  # there are a lot of mismatches between detected faces and celebrities/descriptions
  # but i'm not doing anything with those right now
  for i, f in enumerate(d.faces):
    found_gender = False
    if f.gender == "Male" and not found_gender:
      found_gender = 'man' in x or ' man ' in str(x) or ' man\'s ' in str(x) or 'boy' in x or ' men ' in str(x)
    else:
      assert f.gender == "Female"
      found_gender = 'woman' in x or ' woman ' in str(x)
    if not found_gender or len(d.faces) > 1:
      logger.trace("{}: index #{}, age={}, gender: {} - {}", filename_for_debugging, i, f.age, f.gender, x)

  return x


# noinspection PyUnresolvedReferences
def main():
  book = ['\pagebreak']
  frame: pd.DataFrame = pd.read_excel('book.xlsx')
  for tup in frame.itertuples():
    if tup.year == 'skip':
      continue
    logger.info('{} {}', tup.dir, tup.title)
    book += gen_for_directory('easy/' + str(tup.dir), tup.title, filename=str(tup.dir) + '.md')
    book += ['\pagebreak']

  writelines('book.md', book)
  run_pandoc('book')


def run_pandoc(prefix: str, extra_args=None):
  command = ['pandoc',
             prefix + '.md',
             '--toc', '-N',  # toc with numbering
             'pandoc.yaml',  # metadata
             '-H', 'pandoc.tex',  # header and margin settings
             '-V', 'subparagraph']  # workaround per https://jdhao.github.io/2019/05/30/markdown2pdf_pandoc/
  if extra_args:
    command += extra_args
  command += [
    '-o', prefix + '.pdf'
  ]
  logger.info('running {}', command)
  subprocess.run(command, check=True)


def writelines(outfile: str, values: Iterable[str]) -> str:
  with open(outfile, 'wt', encoding='utf-8') as out:
    lines = 0
    spaces = 0
    chars = 0
    sentences = 0
    for v in values:
      out.write(v)
      out.write('\n\n')
      if v:
        lines += 1
        chars += len(v)
      spaces += v.count(' ')
      sentences += v.count('.')
    out.close()
    logger.info('wrote {} lines to {} ({} words, {} sentences, {} chars)', lines, outfile, spaces, sentences, chars)

  return outfile


def gen_for_directory(directory, header, filename: str = '', window: int = 4, randomize_order: bool = True):
  if not filename:
    filename = directory + '.md'

  assert filename.endswith('.md'), filename
  assert not directory.endswith('/'), directory
  files = glob.glob(directory + '/*.png')
  files.sort()
  assert files, directory
  w4 = gen_one_chapter(header, files, window, randomize_order)
  writelines('.debug-parts/' + filename, w4)
  return w4


global_phrases = Counter()


def gen_one_chapter(header, all_inputs, window, randomize_order, include_removed=False):
  lines = ['# ' + header]

  skip = {"mammal", "person", "land vehicle", "portrait photography", "linedrawing", 'screenshot', 'text', 'font',
          'wearing'}
  all_phrases = Counter()
  keeps = []
  missing = 0
  for f in tqdm(all_inputs):
    d = analyze(f)
    if d is None:
      missing += 1
      continue

    if d.adult.is_adult_content:
      save_class(f, '.debug-class/adult/')
    if d.adult.is_racy_content:
      save_class(f, '.debug-class/racy/')
    if d.adult.is_gory_content:
      save_class(f, '.debug-class/gory/')

    phrases = extract_text(d, f)
    reduced_phrases, removed = _reducer(phrases)

    keep = []
    for p in reduced_phrases:
      if p in skip:
        removed.append(p)
        continue
      q = p[0].upper() + p[1:]
      q = q.replace('et al.', 'and others')
      q = q.replace('christmas', 'Christmas')
      q = q.replace('George Holz', 'Madonna')  # what is this i don't even
      q = q.replace('Qr code', 'QR code')
      q = q.replace('Pc game', 'PC game')
      q = q.replace('Close up', 'Close-up')
      q = q.replace('close up', 'close-up')
      q = q.replace('Cg artwork', 'Computer-generated artwork')
      q = q.replace('Linedrawing', 'Line drawing')
      keep.append(q)

    all_phrases.update(keep)
    global_phrases.update(keep)
    keeps.append((keep, removed))

  for ig in skip:
    assert ig not in all_phrases

  logger.info('{}: most common: {}', header, all_phrases.most_common(20))
  if missing:
    logger.info('{}: missing file count: {} of {}', header, missing, len(all_inputs))

  with open('.debug-lines/' + header + '.json', 'wt') as json_dump:
    json.dump([k[0] for k in keeps], json_dump, indent=2)

  line = 0
  next_at = {}

  for full_line, removed in keeps:
    line = line + 1
    kept = []
    for k in full_line:
      available_at = next_at.get(k, line)
      okay = line >= available_at
      if okay:
        kept.append(k)
        next_at[k] = line + window
      else:
        removed.append(k)
    removed_strike = '~~' + ' '.join(removed) + '~~'
    if not include_removed or not removed:
      removed_strike = ''
    if not kept:
      lines.append('')
      if removed_strike:
        lines.append(removed_strike)
      line = line - 1
      continue

    kept_periods = [k + '.' for k in kept]
    if randomize_order:
      kept_periods = textmods.do_anything(kept_periods)

    t = ' '.join(kept_periods) + ' ' + removed_strike
    t = t.strip()
    lines.append(t)
  return lines


def save_class(f, p):
  os.makedirs(p, exist_ok=True)
  shutil.copy(f, p)


def bulk_downloader():
  globs = ['easy/*/*.png']
  files = (
      get_bulk_files(8, globs) +
      get_bulk_files(4, globs) +
      get_bulk_files(2, globs) +
      get_bulk_files(1, globs))

  seen = set()
  uniques = []
  for f in files:
    if f not in seen:
      uniques.append(f)
      seen.add(f)
  assert len(uniques) < len(files)

  needed = []
  for f in uniques:
    if not fast_isfile('.cache/' + get_file_hash(f) + '.pkl'):
      needed.append(f)

  for f in tqdm(needed):
    a = analyze(f)
    assert a, f

  return True


def get_bulk_files(divisor, globs):
  ret = []
  for g in globs:
    files = glob.glob(g)
    assert files, g
    tmp = [f for i, f in enumerate(files) if i % divisor == 0]
    assert tmp, g
    print(g, len(files), len(tmp))
    ret.extend(tmp)
  return ret


# noinspection PyBroadException
def retry_bulk():
  for attempt in range(300):
    print('attempt #', attempt)
    try:
      success = bulk_downloader()
      return success
    except:
      logger.exception('ignoring')
      time.sleep(61)
      continue


def extract_parent_path(o):
  path = []
  while o:
    path.append(o['object_property'].lower())
    o = o['parent']
  return path


def get_generic_terms_for(word) -> List[str]:
  to_generic = _get_to_generic()
  if word not in to_generic:
    return []
  parent = to_generic[word]
  return [parent] + get_generic_terms_for(parent)


def _get_to_generic():
  return build_hierarchy("2")


@persist_to_file('.cache/hierarchy.json')
def build_hierarchy(version: str):
  # this can take >= 20 seconds to run, so increment version above when there are "a lot" of new json files to process
  logger.info('building version {}', version)
  paths = set()
  to_generic = dict()
  for j in glob.glob('.debug/*.json'):
    with open(j, 'rt') as source:
      obj = jsonpickle.loads(source.read())
    for o in obj.get('objects', []):
      if not o['parent']:
        continue
      p = tuple(extract_parent_path(o))
      if p not in paths:
        logger.info("{}", p)
        paths.add(p)
      for i in range(len(p) - 1):
        child = p[i]
        parent = p[i + 1]
        if child in to_generic:
          assert to_generic[child] == parent, (child, parent, to_generic)
        to_generic[child] = parent

  logger.info('specific terms: {}', len(to_generic))
  return to_generic


for _d in ['.cache', '.debug', '.debug-class', '.debug-lines', '.debug-parts']:
  if not os.path.isdir(_d):
    os.makedirs(_d)

if __name__ == "__main__":
  retry_bulk()
  main()
  if all_celebs:
    logger.info("{}", all_celebs)
  if global_phrases:
    logger.info('global most common: {}', global_phrases.most_common(20))
  logger.info('done')
