class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def study(self):
        print("study.....")

def add(a, b):
    print(a + b)


if __name__ == '__main__':
    ou = Student("ou", 18)
    print(ou.name)
    if hasattr(ou, "score"):
        print(ou.score)
    else:
        print("没有score属性, 设置一个score")
        setattr(ou, "score", 100)

    if hasattr(ou, "score"):
        print(getattr(ou, "score"))

    # delattr(ou, 'name')
    # print(ou.name)

    if hasattr(ou, "study"):
        print("有study_方法")
        ou.study()


    a = add
    a(1, 2)

