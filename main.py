import logging
import os

import click

from ocr_process import OCR
from pipeline import OCRPipeline
from processing import PostProcessor, PreProcessor

DOC_EXT = {'.pdf'}
IMG_EXT = {'.png', '.jpg', 'jpeg'}

log = logging.getLogger('OCR process')
log.setLevel(logging.INFO)


@click.command()
@click.option('-i',
              '--input_path',
              required=True)
@click.option('-o',
              '--output_path',
              required=True)
@click.option('--verbose',
              required=False,
              is_flag=True, type=bool)

def main(output_path, input_path, verbose):
    log.info("Process started")
    file_format = os.path.splitext(input_path)[1]

    if file_format not in IMG_EXT.union(DOC_EXT):
        report = 'InputFormatError: UNSUPPORTED FILE EXTENSION. Supported: .pdf, ' \
            '.png, .jpg or .jpeg but {} found.'.format(file_format)
        log.error(report)
        raise click.BadParameter(report)

    log.info("Checked extension")

    if not verbose:
        logging.basicConfig(
            filename='logs.txt',
            format='%(name)s ... %(asctime)s ... %(levelname)s ... %(message)s',
            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s ... %(asctime)s ... %(levelname)s ... %(message)s',
            datefmt='%H:%M:%S')

    log.info("Created processors")
    pipeline = OCRPipeline(PreProcessor(),
                           OCR(),
                           PostProcessor(), file_format)
    TEXT = pipeline.recognize(input_path)
    log.info("Writing to the output file...")

    try:
        with open(output_path, 'w') as f:
            f.writelines(TEXT)
    except FileNotFoundError as e:
        report = 'Output file not found!'
        log.error(report)

    log.info("Finished!")


if __name__ == '__main__':
    main()
