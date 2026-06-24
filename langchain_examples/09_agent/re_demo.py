class Student():
    id: int
    name: str

    def __init__(self, id, name):
        self.id = id
        self.name = name


if __name__ == '__main__':
    jack = Student(1, "Jack")
    if not hasattr(jack, "age"):
        setattr(jack, "age", "22")
        print(jack.age)