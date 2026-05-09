from collections.abc import Iterable
from itertools import combinations
from functools import lru_cache
from typing import List, Tuple, Union


@lru_cache(maxsize=65536)
def calculate_char(x: int,y: int,length: int)->int:
    if x < y:x,y = y,x
    return min(x-y,length+y-x)

@lru_cache(maxsize=256)
def two_side_mod(x,modp):
    return min(x,modp-x)

@lru_cache(maxsize=65536)
def calculate_characteristic(ruler: 'CircularRuler', cnt_rows: int) -> Tuple[set]:
    result = [ruler.bins]
    shifted_ruler = ruler
    for _ in range(1,cnt_rows):
        shifted_ruler = shifted_ruler << 1
        result.append(shifted_ruler-ruler)
    return tuple(result)

def check_collide(charA:List[set],charB:List[set]):
    for sA,sB in zip(charA,charB):
        if not sA.isdisjoint(sB):
            return True
    return False

def add_char(charA:List[set],charB:List[set]):
    return [sA.union(sB) for sA,sB in zip(charA,charB)]

def find_viable_combinations(cur_characteristic,remaining_choices,cap,cnt_rows)->Union[List['CircularRuler'], bool]:
    if cap == 0:
        return []
    for idx,choices in enumerate(remaining_choices,start=1):
        cur_char = calculate_characteristic(choices,cnt_rows)
        if check_collide(cur_characteristic,cur_char):
            continue
        res = find_viable_combinations(add_char(cur_characteristic,cur_char),remaining_choices[idx:],cap-1,cnt_rows)
        if res != False:
            res.append(choices)
            return res
    return False

class CircularRuler:
    __slots__ = ('length', 'markers', '_binary_repr', '_is_golomb', '_bins')
    def __init__(self,length:int,markers:Iterable=None):
        self.length:int = length
        self.markers:set = {i%self.length for i in markers} if markers else set()
        self._binary_repr:int = None
        self._is_golomb:bool = None
        self._bins:set = set()

    def add(self,element:int):
        self.markers.add(element%self.length)
        self._is_golomb = None

    @property
    def bins(self)->set:
        if not self._bins:
            self.check_golomb()
        return self._bins

    @property
    def binary_repr(self) -> int:
        if self._binary_repr is None:
            self._binary_repr = sum(1<<i for i in self.markers)
        return self._binary_repr

    def __int__(self) -> int:
        return self.binary_repr

    def __contains__(self,element):
        return element in self.markers

    def check_golomb(self) -> bool:
        if self._is_golomb is None:
            seen_distances = set()
            for x,y in combinations(self.markers,2):
                distance = calculate_char(x,y,self.length)
                if distance in seen_distances:
                    self._is_golomb = False
                    break
                seen_distances.add(distance)
            else:
                self._is_golomb = True
                self._bins = seen_distances
        return self._is_golomb 

    def __repr__(self):
        return '+'.join(f'x^{marker}' for marker in sorted(self.markers))

    def __lshift__(self,value: int) -> 'CircularRuler':
        if not self.markers:
            return CircularRuler(self.length)
        new_markers = {(i << value) for i in self.markers}
        new_ruler = CircularRuler(self.length, new_markers)
        new_ruler._is_golomb = self._is_golomb
        #new_ruler._bins = {f(i<<1) for i in self._bins}
        return new_ruler

    def visualize(self,writer_obj,x,y,limit=None):
        if not limit:
            limit = self.length
        else:
            limit = min(self.length,limit)
        for nx in range(limit):
            for marker in self.markers:
                new_y = (nx+marker)%self.length
                if new_y >= limit: continue
                writer_obj[x+nx,y+new_y] = 1
        return writer_obj

    def __len__(self):
        return len(self.markers)

    def __sub__(self,another_ruler: 'CircularRuler') -> set:
        result = set()
        f = lambda x:x+self.length if x < 0 else x
        for x in self.markers:
            for y in another_ruler.markers:
                result.add(f(x-y)) # x1-y1 = x2-y2 will not happen, otherwise x1-x2 = y1-y2, for 2 golomb rulers
        return result

