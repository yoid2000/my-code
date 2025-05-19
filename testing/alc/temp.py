
def _select_evenly_distributed_values(sorted_list: list, num_prc_measures: int) -> list[float]:
    '''
    This splits the confidence values into num_prc_measures evenly spaced values.
    Note by the way that this is being used on the attack confidence values.
    '''
    if len(sorted_list) <= num_prc_measures:
        return sorted_list
    selected_values = [sorted_list[0]]
    step_size = (len(sorted_list) - 1) / (num_prc_measures - 1)
    for i in range(1, num_prc_measures - 1):
        index = int(round(i * step_size))
        selected_values.append(sorted_list[index])
    selected_values.append(sorted_list[-1])
    return selected_values


my_list = list(range(1, 100))
# sort my_list
my_list = sorted(my_list, reverse=True)
selected = _select_evenly_distributed_values(my_list, 21)
print(len(selected))