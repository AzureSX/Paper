interact是三维交互数据

user-item-ratings

```python
item_counts = get_all_item_counts(interact, item_list)
```

将 item-id list化

```python
item_list = interact['itemid'].tolist()
```



```python
item_counts = get_all_item_counts(interact, item_list)
```

```python
# data = interact
# item_list = item_list
def get_all_item_counts(data, item_list):
    item_counts = parallelize_on_rows(data, partial(get_item_count, deepcopy(item_list)))
    return item_counts
```

要获取 item_counts，由 parallelize_on_rows 函数返回

```python
# data = data = interact
# func = partial(get_item_count, deepcopy(item_list)) = 
# num_of_processes = core
def parallelize_on_rows(data, func, num_of_processes=core):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)
```

```python
def parallelize(data, func, num_of_processes=core):
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data_list = pool.map(func, data_split)
    data = pd.concat(data_list)
    return data
```

