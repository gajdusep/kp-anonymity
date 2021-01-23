v = False


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
