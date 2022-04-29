from abc import ABC
from logging import getLogger

import numpy as np
import pytesseract

log = getLogger('OCR process')


class OCR(ABC):

    """
    OCR by PyTesseract lib
    """

    def transform(self, image: np.array) -> str:
        """
        :param image: input image
        :return: OCR text
        """
        return pytesseract.image_to_string(image)
