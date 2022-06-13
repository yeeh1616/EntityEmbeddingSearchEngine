if __name__ == '__main__':
    intColor = 10000
    a = "#{:06x}".format(0xFFFFFF & intColor)
    print(a)