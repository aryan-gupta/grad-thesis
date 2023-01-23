


class A:
    def __init__(self):
        self.state = 'A'
        pass

    def pas(self):
        print(self.state)

    def pa(self):
        print("helloA")

    def update(self):
        self.__class__ = B
        self.__init__(self)


class B(A):
    def __init__(self, a):
        self.state = 'B'
        pass

    def pb(self):
        print("helloB")

    def pabs(self):
        print(self.state)

    # def promote(self):
    #     # self.state = 'B'
    #     pass



a = A()
a.pa()
a.pas()
a.update()
# a = a.update()
a.pb()
a.pa()
a.pabs()
