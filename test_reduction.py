import itertools
import json
import pprint
from collections import Iterable, defaultdict
from typing import List

import pytest

from cloud_vision import reduce, _reducer, is_overlap, is_overlap_or_exact, get_generic_terms_for


def test_0():
  orig = ['a black rectangle with a black background', 'dark', 'night sky', 'space', 'star', 'darkness', 'constellation', 'astronomy',
          'astronomical object',
          'universe', 'sky', 'outer space', 'galaxy', 'night', 'nature']

  k, r = _reducer(orig)
  assert k == ['a black rectangle with a black background', 'night sky', 'star', 'darkness', 'constellation', 'astronomy',
               'astronomical object',
               'universe', 'outer space', 'galaxy', 'nature']
  assert r == ['dark', 'space', 'sky', 'night']


def test_1():
  assert reduce(['test', 'test']) == ['test']


def test_2():
  assert reduce(['a man', 'man']) == ['a man']


def test_3():
  assert reduce(['a woman', 'man']) == ['a woman', 'man']


def test_4():
  assert reduce(['this is a journey', 'night sky', 'night', 'sky']) == ['this is a journey', 'night sky']


def test_5():
  assert reduce(['A woman dancing', 'woman', 'dancing']) == ['A woman dancing']
  assert reduce(['Madonna is dancing', 'Madonna', 'dancing']) == ['Madonna is dancing']


def test_6():
  assert reduce(['a group of dogs in a mall', 'dog']) == ['a group of dogs in a mall']


def test_7():
  assert reduce(['clouds', 'cloud', 'cloudy', 'A cloudy sky full of clouds']) == ['A cloudy sky full of clouds']


def test_8():
  s = "A person working in a kitchen. Standing. Preparing. Cooking. Table. Kitchen appliance. Stove. Home appliance. Countertop. Small appliance. Shelf. Food. Appliance"
  phrases = s.lower().split('. ')
  expected = ['a person working in a kitchen',
              'standing',
              'preparing',
              'cooking',
              'table',
              'kitchen appliance',
              'stove',
              'home appliance',
              'countertop',
              'small appliance',
              'shelf',
              'food']

  assert reduce(phrases) == expected
  assert len(phrases) - 1 == len(expected)


def test_9():
  assert reduce(['a man and woman kissing', 'kiss']) == ['a man and woman kissing']


def test_10():
  orig = ['a man in a suit and tie', 'person', 'man', 'indoor', 'standing', 'suit', 'person',
          'wrist', 'blazer', 'clothing', 'finger', 'collar', 'joint',
          'thumb', 'holding hands', 'outerwear', 'wearing', 'suit', 'hand', 'person']
  k, r = _reducer(orig)
  assert r == ['man', 'suit', 'person', 'suit', 'hand', 'person']


def test_overlaps():
  assert not is_overlap(['dark', 'darkness'], 'darkness')
  assert is_overlap(['dark', 'covered in darkness'], 'darkness')
  assert is_overlap_or_exact(['dark', 'covered in darkness'], 'darkness')

  assert not is_overlap(['dark', 'covered in darkness'], 'dark')


def test_11():
  orig = ['dark', 'darkness']
  k, r = _reducer(orig)
  assert k == ['darkness']


def test_12():
  # electric_guitar -> guitar -> string_instrument -> musical_instrument
  assert get_generic_terms_for('guitar') == ['string instrument', 'musical instrument']
  orig = ['a person playing an electric guitar', 'electric guitar', 'guitar', 'string instrument', 'musical instrument']

  assert set(itertools.chain.from_iterable(get_generic_terms_for(w) for w in orig)) == {'guitar', 'musical instrument', 'string instrument'}

  k, r = _reducer(orig)
  assert k == [orig[0]]
  assert r == orig[1:]


def test_13():
  orig = ['bath', 'bathroom', 'bathtub', 'tub']
  k, r = _reducer(orig)
  assert k == ['bathroom', 'bathtub']
  orig = ['road', 'roadway', 'way']
  k, r = _reducer(orig)
  assert k == ['roadway']


def to_word_stream(lines: List[List[str]]) -> Iterable[str]:
  for words in lines:
    yield from words
    yield ''


def to_paragraphs(words: List[str]) -> Iterable[List[str]]:
  for new_paragraph, g in itertools.groupby(words, lambda x: x == ''):
    if new_paragraph:
      continue
    yield list(g)


def test_ws():
  orig = [['a', 'b'], ['c'], ['d']]
  actual = list(to_word_stream(orig))
  expected = ['a', 'b', '', 'c', '', 'd', '']
  assert actual == expected
  rt = list(to_paragraphs(actual))
  assert rt == orig


if __name__ == '__main__':
  pytest.main()
