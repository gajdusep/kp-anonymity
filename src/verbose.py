v = False
d = False


def getverbose():
    return v


def setverbose():
    global v
    v = True
    return


def unsetverbose():
    global v
    v = False
    return


def verbose(text: str):
    if v:
        print(text)
    return


def getdebug():
    return d


def setdebug():
    global d
    d = True
    return


def unsetdebug():
    global d
    d = False
    return


def debug(text: str):
    if d:
        print(text)
    return
