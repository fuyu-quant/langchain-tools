```python
def multiples_of_three(query: str) -> str:
    'Outputs multiples of 3 up to 20. The input is a single string. For example, if you want to get multiples of 3 up to 20, the input is "20".'
    limit = int(query)
    result = [str(i) for i in range(3, limit+1, 3)]
    return ", ".join(result)
```