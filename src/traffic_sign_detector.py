import argparse



'''
    Traffic Sign Detector: Main entry point of the application
'''


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ITS Project: Spring-2018")

    parser.add_argument(
        '--file',
        default="null",
        help="Input video file"
    )

    args = parser.parse_args()
    main(args)
