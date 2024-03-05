
def simple_inequalities_solver(counts_out, N_delta):
    # This takes a simple set of inequalities and produces the
    # max and min possible solutions
    min_val = None
    max_val = None

    for Y in counts_out:
        if max_val is None:
            max_val = Y + N_delta
        else:
            max_val = min(max_val, Y + N_delta)

        if min_val is None:
            min_val = Y - N_delta
        else:
            min_val = max(min_val, Y - N_delta)

    return min_val, max_val


if __name__ == "__main__":
    counts_out = [31, 38, 42, 41]
    N_delta = 10  
    
    min_X, max_X = simple_inequalities_solver(counts_out, N_delta)
    print(f"The value of X lies between {min_X} and {max_X}.")