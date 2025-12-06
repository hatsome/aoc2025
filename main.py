import os
import pathlib
import functools
import textwrap
import itertools
import operator
from dataclasses import dataclass

def main(day: str, part: str):
    input_file = pathlib.Path('inputs') / f"{day + "e" if part == "0" else day}" 
    with open(input_file, 'r') as f:
        input = f.read()
    
    func = globals()[f"day_{day}"]
    output = func(input, part)
    print(f"Output:\n{output}")

def day_1(input: str, part: str):
    rotation_list = map(lambda x: int({"L": "-", "R": "+"}[x[0]] + x[1:]), input.splitlines())
    if part == "1":
        prev = 50
        position_list = [prev := (rot + prev) % 100 for rot in rotation_list]
        return position_list.count(0)
    if part == "0" or part == "2":
        prev = (0, 50)
        def cnt_pos(rot, pos):
            cnt, n_pos = divmod(rot + pos, 100)
            cnt = abs(cnt)
            if pos == 0 and rot < 0:
                cnt -= 1
            elif n_pos == 0 and rot < 0:
                cnt += 1
            return cnt, n_pos
        cnt_pos_list = [prev := cnt_pos(rot, prev[1]) for rot in rotation_list]
        return functools.reduce(lambda x, y: (abs(x[0]) + abs(y[0]), 0), cnt_pos_list)[0]

def day_2(input: str, part: str):
    def is_invalid(id: str):
        if part == "1":
            if len(id) % 2 == 1:
                return False
            return id[:len(id)//2] == id[len(id)//2:]
        if part == "0" or part == "2":
            for repeats in range(2, len(id) +1):
                if len(id) % repeats == 1:
                    continue
                if len(set(textwrap.wrap(id, len(id)//repeats))) == 1:
                    return True
            return False
    ranges = map(lambda x: tuple(map(lambda y: int(y), x.split("-"))),input.split(","))
    invalid_sum = 0
    for start, end in ranges:
        for id in range(start, end + 1):
            if is_invalid(str(id)):
                invalid_sum += id
    return invalid_sum

def day_3(input: str, part: str):
    def get_batteries(bank: str, num: int):
        if num <= 0:
            return max(bank)
        else:
            idx_0 = bank.index(max(bank[:-num]))
            return bank[idx_0] + get_batteries(bank[idx_0+1:], num - 1)
    banks = input.splitlines()
    sum = 0
    for bank in banks:
        if part == "1":
            num = 1
        elif part == "0" or part == "2":
            num = 11
        sum += int(get_batteries(bank, num))
    return sum

def day_4(input: str, part: str):
    diagram = input.splitlines()
    nadjacant = {}
    for row, line in enumerate(diagram):
        for col, pos in enumerate(line):
            if pos == "@":
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if not x == y == 0:
                            idx = (row + y, col + x)
                            nadjacant[idx] = nadjacant.get(idx, 0) + 1

    def remove_step():
        nremoved_step = 0
        nonlocal nadjacant
        next_nadjacant = nadjacant.copy()
        for row, line in enumerate(diagram):
            for col, pos in enumerate(line):
                if pos == "@" and nadjacant.get((row, col), 0) < 4:
                    diagram[row] = diagram[row][:col] + "." + diagram[row][col+1:]
                    nremoved_step += 1
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if not x == y == 0:
                                idx = (row + y, col + x)
                                next_nadjacant[idx] = next_nadjacant.get(idx, 0) - 1
        nadjacant = next_nadjacant
        return nremoved_step

    if part == "1":
        return remove_step()
    elif part == "0" or part == "2":
        nremoved = 0
        while True:
            nremoved_step = remove_step()
            if nremoved_step > 0:
                nremoved += nremoved_step
            else:
                break
        return nremoved

def day_5(input: str, part: str):
    @dataclass
    class FreshRange():
        start: int
        stop: int

        def overlaps(self, other: FreshRange):
            if other.start <= self.stop <= other.stop:
                return True
            if other.start <= self.start <= other.stop:
                return True
            return False
        
        def absorb(self, other: FreshRange):
            start = None
            if self.start <= other.start:
                start = self.start
            else:
                start = other.start
            stop = None
            if self.stop >= other.stop:
                stop = self.stop
            else:
                stop = other.stop
            self.start = start
            self.stop = stop
        
        def inside(self, value: int):
            return self.start <= value <= self.stop
        
        def count(self):
            return self.stop - self.start + 1
        
        def __repr__(self):
            return f"{self.start}-{self.stop}"

    ranges, ids = map(lambda x: x.splitlines(), input.split("\n\n"))
    fresh_ranges = []
    for range in ranges:
        start, stop = map(lambda x: int(x), range.split("-"))
        fresh_ranges.append(FreshRange(start, stop))
    fresh_ranges.sort(key=lambda x: x.start)
    
    combined_ranges = [fresh_ranges[0]]
    for fresh_range in fresh_ranges[1:]:
        if fresh_range.overlaps(combined_ranges[-1]):
            combined_ranges[-1].absorb(fresh_range)
        else:
            combined_ranges.append(fresh_range)
    
    if part == "1":
        count = 0
        for id in map(lambda x: int(x), ids):
            is_fresh = any(map(lambda x: x.inside(id), combined_ranges))
            if is_fresh:
                count += 1
        return count
    elif part == "0" or part == "2":
        return sum(map(lambda x: x.count(), combined_ranges))


def day_6(input: str, part: str):
    if part == "1":
        return sum(functools.reduce(op, nums) for op, nums in zip(({"*": operator.mul,"+": operator.add}[op] for op in input.splitlines()[-1].split()), zip(*(map(int, line.split()) for line in input.splitlines()[:-1]))))
    elif part == "0" or part == "2":
        return sum(functools.reduce(op, (int(num) for num in nums)) for op, nums in zip(({"*": operator.mul,"+": operator.add}[op] for op in input.splitlines()[-1].split()), (list(group) for is_num, group in itertools.groupby(["".join(sym).strip() for sym in zip(*(itertools.chain(line) for line in input.splitlines()[:-1]))], key=str.isdigit) if is_num)))

if __name__ == '__main__':
    day = input('Input puzzle [day]: ')
    part = input('Input puzzle [part]: ')
    main(day, part)