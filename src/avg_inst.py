def avg_inst(x):
    return {
        < 200 : 1,
        > 200 : 2,
        }.get(x, 9) 