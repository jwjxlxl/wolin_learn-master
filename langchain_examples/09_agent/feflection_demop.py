'''

    Python中的反射机制：在运行时可以动态的获取对象的信息，包括属性和方法
    也可以动态的设定对象的属性，方法，删除对象的属性，方法
'''

class Student():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def study(self, subject: str):
        print(f"{self.name}正在学习{subject}")


if __name__ == '__main__':
    hongyu = Student("张鸿宇", 25)
    # 在运行时获取对象属性，做判断，增强程序的健壮性
    if hasattr(hongyu, "height"):
        print("鸿宇的身高是：", hongyu.height)
    else:
        setattr(hongyu, "height", 175)
        print(f"鸿宇没有身高属性, 新设定了身高{hongyu.height}")
    # print(hongyu.heigth)


    hongyu.study("python")
    if hasattr(hongyu, "playgame"):
        hongyu.playgame()
    else:
        print("鸿宇没有打游戏方法")

    n = getattr(hongyu, "name")   # 等同于 hongyu.name
    print(n)