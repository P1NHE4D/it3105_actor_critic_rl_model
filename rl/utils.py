class DefaultValueTable(dict):

    def __init__(self, val_func, **kwargs):
        super().__init__(**kwargs)
        self.val_func = val_func

    def __getitem__(self, item):
        if item not in super().keys():
            super().__setitem__(item, self.val_func())
        return super().__getitem__(item)
