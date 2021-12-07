

PROPORTION_CONSTANTS = {
    # Hand
    "se":  0.54,
    "ew":  0.5333333333333333,
    # Legs
    "hk":  0.812857142857143,
    "ka":  0.7271428571428571,
    # Between hips
    'hh': 0.37833333333333335,
    # Between shoulders
    'ss': 0.64
}

PROPORTIONS_INDX = {
    # Hand | left and right
    "se": [(4, 6), (5, 7)],
    "ew": [(6, 8), (7, 9)],
    # Legs | left and right
    "hk": [(10, 12), (11, 13)],
    "ka": [(12, 14), (13, 15)],
    # Between hips | only one side
    'hh': [(10, 11)],
    # Between shoulders | only one side
    'ss': [(4, 5)]
}
