class PandaSdx:
    class _Iloc:
        def __getitem__(self, item):
            print(item)

    def __init__(self):
        self.iloc = self._Iloc()

    def __getitem__(self, item):
        print(item)

# Create a PandaSdx object
psdx = PandaSdx()

# Test the new behavior
psdx['x', 'y', 'z']
psdx.iloc[1:2, [3, 4]]
