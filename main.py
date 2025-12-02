import pathlib
import functools
import textwrap

def main(day: str, part: str):
    input_file = pathlib.Path('inputs') / f"{day}_{part}"
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
    if part == "0" or "2":
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
        if part == "0" or "2":
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


if __name__ == '__main__':
    day = input('Input puzzle [day]: ')
    part = input('Input puzzle [part]: ')
    main(day, part)
