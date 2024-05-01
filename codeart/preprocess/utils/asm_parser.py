

def ispurenumber(number):
    if number[0] == '+' or number[0] == '-':
        number = number[1:]
    # whether every char is digit
    for i in range(len(number)):
        if str.isdigit(number[i]):
            continue
        else:
            return False
    return True


def isaddr(number):
    return number[0] == '[' and number[-1] == ']'


def ishexnumber(number):
    if number[0] == '+' or number[0] == '-':
        number = number[1:]
    if number[-1] == 'h':
        for i in range(len(number)-1):
            if str.isdigit(number[i]) or (number[i] >= 'A' and number[i] <= 'F'):
                continue
            else:
                return False
    else:
        return False
    return True
