

class Group:

    def __init__(self):
        self.some_value = 2        

    @staticmethod
    def some_static_function():
        print("something")

    def some_non_static_function(self):
        print(self.some_value)


if __name__ == "__main__":
    Group.some_static_function()

    my_group = Group()
    my_group.some_non_static_function()
