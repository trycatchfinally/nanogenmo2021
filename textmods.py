import inspect
import random
from typing import List, Dict


def do_nothing(words):
  return words


def sort_by_spaces_then_length(words):
  return sorted(words, key=lambda x: (-x.count(' '), len(x)))


def sort_by_spaces_then_alpha(words):
  return sorted(words, key=lambda x: (-x.count(' '), x))


def sort_shuffle(words):
  if len(words) == 1:
    return words
  before = list(words)
  while before == words:
    # print('shuffling')
    random.shuffle(words)
  return words


def sort_reverse(words):
  return list(reversed(words))


def join_until_1(words):
  return _join_until_n(words, 2)


def join_until_2(words):
  return _join_until_n(words, 2)


def join_until_3(words):
  return _join_until_n(words, 3)


def join_until_4(words):
  return _join_until_n(words, 4)


def join_until_halved(words):
  return _join_until_n(words, len(words) // 2)


def _join_until_n(words: List[str], n: int) -> List[str]:
  if len(words) <= n:
    return words

  first, second = random.sample(words, 2)
  words.remove(first)
  words.remove(second)
  first = first.lower().rstrip(',. ')
  second = second.lower().rstrip(',. ')
  c = first + ', ' + second + '.'
  c = c[0].upper() + c[1:]
  words.append(c)

  return _join_until_n(words, n)


def join_single_front(words):
  return _join_single_words(words, True)


def join_single_end(words):
  return _join_single_words(words, False)


def _join_single_words(words, at_beginning):
  if len(words) == 1:
    return words
  singles = [w.lower().rstrip('.') for w in words if ' ' not in w]
  if not singles:
    return words
  ret = []
  ret.extend(p for p in words if ' ' in p)
  c = ', '.join(singles) + '.'
  c = c[0].upper() + c[1:]
  if at_beginning:
    ret.insert(0, c)
  else:
    ret.append(c)
  return ret


def do_everything(words):
  current_module = __import__(__name__)
  seen = set()
  for name, func in inspect.getmembers(current_module, inspect.isfunction):
    if name in {'do_everything', 'do_anything'}:
      continue
    if name.startswith('_') or '_until_' in name:
      continue
    copy = list(words)
    values = func(copy)
    assert isinstance(values, list), (name, values)
    v = ' '.join(values)
    if v in seen:
      continue
    if v.startswith('.'):
      assert False, (name, words, values)
    seen.add(v)
    yield values


def do_anything(words) -> List[str]:
  for w in words:
    assert w, words
  return random.choice(list(do_everything(words)))


_appearance_lookup: Dict[str, int] = dict()


def sort_global_appearance(words) -> List[str]:
  terms = []
  for w in words:
    k = w.lower()
    if k not in _appearance_lookup:
      _appearance_lookup[k] = len(_appearance_lookup)
    v = _appearance_lookup[k]
    terms.append((v, w))
  terms.sort()
  return [t[1] for t in terms]


if __name__ == "__main__":
  example = ['A group of people with their hands up in the air.', 'Television.', 'Flat.', 'Display.', 'Music.']

  for opts in do_everything(example):
    print(' '.join(opts))

  print(do_anything(example))
  print(sort_global_appearance(['First', 'Second', 'Third']))
  print(sort_global_appearance(['Third', 'Second', 'New Last', 'First']))
  print(_appearance_lookup)
