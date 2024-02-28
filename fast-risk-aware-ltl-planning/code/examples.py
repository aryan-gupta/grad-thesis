
from enum import Enum
from PIL import Image
import numpy as np

import global_vars as gv

class Examples(Enum):
    NORMAL_ENV_EX = 0
    ASYNC_ENV_EX = 1
    CHOIC_ENV_EX = 2
    CHOIB_ENV_EX = 3
    EXAMPLES_NUM = 4

class EnvironmentCreatorExample:
    def __init__(self, example):
        """
            The num parameter is for the choice in which environment example to create:
            1 - A basic 5x5 environment with 4 risk spots (corners of the
                3x3 center concentric square) and 4 targets (corners)
                ________
                |T    T|
                | X  X |
                |      |
                | X  X |
                |T    T|
                ________

            2 - Same thing as 1 but the bottom left target is moved up one cell, this
                creates an assemetry that allows the choice for one path to be shorter
                than the other
                ________
                |T    T|
                | X  X |
                |      |
                |TX  X |
                |     T|
                ________

            3 - This example uses the same thing as 1 but examplifies task switching
                the task is A -> loop:(B -> C [-> B]) -> D. but on the second iteration
                of the loop, the task switching kicks in and orders the bot to head
                straight to D

        """
        self.img = self.create(example)

    def create(self, example):
        array = self.create_array(example)
        nparray = self.convert_to_rgb_nparray(array)
        img = self.create_img(nparray)
        return img

    def convert_to_rgb_nparray(self, array):
        nparray = []

        for r, row in enumerate(array):
            new_row = []
            for c, ele in enumerate(row):
                if   ele == ' ': new_row.append((   0,   0,   0))
                elif ele == 'X': new_row.append(( 255,   0,   0))
                else:
                    green = [key for key, value in gv.CHAR_COLOR_MAP.items() if value == ele][0]
                    new_row.append((   0, green,   0))
            nparray.append(new_row)
        print(nparray)
        return np.array(nparray, dtype=np.uint8)


    def create_img(self, nparray):
        img = Image.fromarray(nparray)
        img = img.resize((32*5, 32*5), resample=Image.Resampling.NEAREST)
        return img

    def create_array(self, example):
        array = self.create_base()

        array[0][0] = gv.START_CELL_CHAR
        if example == Examples.NORMAL_ENV_EX:
            array[0][4] = 'B'
            array[4][0] = 'C'
            array[4][4] = gv.END_CELL_CHAR
        elif example == Examples.ASYNC_ENV_EX:
            array[0][4] = 'B'
            array[3][0] = 'C'
            array[4][4] = gv.END_CELL_CHAR
        elif example == Examples.CHOIC_ENV_EX:
            array[0][4] = 'B'
            array[4][0] = 'C'
            array[4][2] = gv.END_CELL_CHAR
        elif example == Examples.CHOIB_ENV_EX:
            array[0][4] = 'B'
            array[4][0] = 'C'
            array[2][4] = gv.END_CELL_CHAR
        else:
            raise Exception("wrong example or example not specified")

        return array

    def create_base(self):
        return [
            [ ' ', ' ', ' ', ' ', ' ' ],
            [ ' ', 'X', ' ', 'X', ' ' ],
            [ ' ', ' ', ' ', ' ', ' ' ],
            [ ' ', 'X', ' ', 'X', ' ' ],
            [ ' ', ' ', ' ', ' ', ' ' ]
        ]

    def show(self):
        self.img.show()

    def save(self, name):
        self.img.save(name)



if __name__ == "__main__":
    e = EnvironmentCreatorExample(Examples.NORMAL_ENV_EX).save('../maps/normal-example.bmp')
    e = EnvironmentCreatorExample(Examples.ASYNC_ENV_EX).save('../maps/async-example.bmp')
    e = EnvironmentCreatorExample(Examples.CHOIC_ENV_EX).save('../maps/choic-example.bmp')
    e = EnvironmentCreatorExample(Examples.CHOIB_ENV_EX).save('../maps/choib-example.bmp')
