# Use this script to add all your predictive model solution.
# You can take advantge of all the libraries that you prefer.


def predict(inputData):
    """This is the function that will be used to test your predictive model.

    It will receive an input data which consist of 168 (1 week == 24h * 7 days)
    lists with the date, hour and NO2 values.

    Something like: [['2023-01-01', 1, 62.0], ...,  ['2023-01-07', 24, 60.0]]

    This function must return a list of 24 values with the NO2 prediction for
    each hour of the next day.
    """
    return [inputData[-1, 2]] * 24
