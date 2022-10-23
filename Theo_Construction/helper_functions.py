def remove_brackets(string):
    result = ''
    for i in string:
        if i == '(':
            break
        result = result + i
    return result

def country_map(Course):
    if Course[-1] == ')':
        if Course[-4] == '(':
            return Course[-3:-1]
        else:
            return Course[-4:-1]
    else:
        return 'GB'