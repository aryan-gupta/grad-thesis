


class A:
    def __init__(self):
        pass

    def pa(self):
        print("helloA")

    def update(self):
        u = B()
        # self = u
        return u


class B(A):
    def __init__(self):
        pass

    def pb(self):
        print("helloB")



a = A()
a.pa()
# a.update()
a = a.update()
a.pb()
a.pa()
