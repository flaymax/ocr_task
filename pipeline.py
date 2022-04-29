import time
from abc import ABC
from logging import getLogger
from tqdm import tqdm

import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

IMG_EXT = {'.png', '.jpg', 'jpeg'}

log = getLogger('OCR process')


class OCRPipeline(ABC):

    """ Basic pipeline for OCR image recognition by PyTesseract lib """

    def __init__(self, preprocessor, ocr, postprocessor, file_ext):
        """
        :param preprocessor: preprocessor class
        :param ocr: OCR class to recognize characters from input file
        :param postprocessor: postprocessor class
        :param file_ext: extension of the input file
        """

        self.preprocessor = preprocessor
        self.ocr = ocr
        self.postprocessor = postprocessor
        self.file_ext = file_ext

    def recognize(self, input_path: str) -> list:
        """
        :param input_path: path to the image file
        :return: recognized text
        """

        if self.file_ext in IMG_EXT:
            log.info('Image Pipeline selected')
            img = cv2.imread(input_path)
            start = time.time()
            preprocessed_image = self.preprocessor.transform(img)
            log.info('Starting OCR.')
            predicted_text = self.ocr.transform(preprocessed_image)
            processed_text = self.postprocessor.transform(predicted_text)
            end = time.time()
            log.info('OCR done! OCR took {} sec.'.format(round(end - start,2)))
            return processed_text

        else:

            log.info('PDF Pipeline selected')
            images = []
            text = []
            try:
                images = convert_from_path(input_path)
            except PDFInfoNotInstalledError as e:
                log.error('PDF info not installed, install poppler')
            log.info('Got {} images out of PDF'.format(len(images)))
            start = time.time()
            log.info('Starting OCR.')
            for i in tqdm(range(len(images)), ncols=90, desc="OCR process"):
                image = self.preprocessor.transform(np.array(images[i]))
                pred_text = self.ocr.transform(image)
                text.append(pred_text)
            end = time.time()
            log.info('OCR done! OCR took {} sec.'.format(round(end - start, 2)))
            return text
