import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process China economic data")
    parser.add_argument(
        "-i", "--input-file", default="china_data_raw.md", help="Input file name"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=1/3, help="Capital share parameter"
    )
    parser.add_argument(
        "-o", "--output-file", default="china_data_processed", help="Base name for output files"
    )
    parser.add_argument(
        "-k", "--capital-output-ratio", type=float, default=3.0,
        help="Capital-to-output ratio for base year (2017)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2025,
        help="Last year to extrapolate/process (default: 2025)"
    )
    return parser.parse_args()
