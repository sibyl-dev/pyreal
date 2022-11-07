class Data:
    def __init__(self, data, **kwargs):
        self.data = data
        self.tags = kwargs
        print(self.tags)

    def add_tags(self, tag_name, tag_values):
        self.tags[tag_name] = tag_values

    def get(self):
        return self.data

    def get_tag(self, tag_name):
        return self.tags[tag_name]


