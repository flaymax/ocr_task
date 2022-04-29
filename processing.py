import math
import re
from abc import ABC
from logging import getLogger

import cv2
import numpy as np
from deskew import determine_skew

log = getLogger('OCR process')


class PreProcessor(ABC):

    """ Basic image preprocessor class """

    def transform(self, image: np.array,background=(255, 255, 255)) -> np.array:
        """
        :param image: input image in preprocessing
        :param background: background colour for rotation (default=white)
        :return: input image after rotation if needed
        """
        image = self.thresholding(image)
        rotated_image = self.rotation(image,
                                      determine_skew(image),
                                      background=background
                                      )
        return rotated_image

    @staticmethod
    def rotation(image, angle, background):

        """
        :param image: image
        :param angle: roatation angle
        :param background: background colour for rotation (default=white)
        :return: rotated image
        """

        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height
                    ) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width
                     ) + abs(np.cos(angle_radian) * old_height)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(
            image, rot_mat, (int(round(height)), int(round(width))),
            borderValue=background
        )

    @staticmethod
    def thresholding(image: np.array) -> np.array:
        """
        :param image: image
        :return: thresholded image
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(
            grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


class PostProcessor(ABC):

    """ Basic image postprocessor class """

    def transform(self, text: str):

        """
        :param text: recognized text
        :return: postprocessed text
        """

        log.info('Deleting some mis‐recognition symbols')
        rx = re.compile(r'(.)\1{2,}')
        cleaned = []

        for line in text.split('\n'):
            cleaned_line = []
            for word in line.split(' '):
                if word != '':
                    if not rx.search(word):
                        cleaned_line.append(word)
                    else:
                        cleaned_line.append('...')
                else:
                    cleaned_line.append('\n')
            cleaned_line.append('\n')
            cleaned.append(' '.join(cleaned_line))

        return cleaned
